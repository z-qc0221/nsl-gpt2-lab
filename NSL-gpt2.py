import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

kv_cache = []
kv_cache_length = 0
kv_cache_capacity = 0


def reset_kv_cache(n_layer, n_head, max_seq_len=1024):
    """为每一层创建延迟分配的、按全部 attention head 组织的 KV Cache。"""
    global kv_cache, kv_cache_length, kv_cache_capacity
    # 优化：每层只维护一个 [n_head, max_seq_len, head_dim] 的 K/V Tensor。
    # 相比原来每个 head 单独 torch.cat，后续可以一次写入全部 head。
    kv_cache = [[None, None] for _ in range(n_layer)]
    kv_cache_length = 0
    kv_cache_capacity = max_seq_len

def clear_kv_cache():
    global kv_cache, kv_cache_length, kv_cache_capacity
    kv_cache = []
    kv_cache_length = 0
    kv_cache_capacity = 0

def kv_cache_len():
    return kv_cache_length


# 投机推理会在草稿模型和目标模型之间切换。缓存采用预分配 Tensor 后，
# 检查点只需保存 buffer 引用与有效长度；不复制 K/V 数据也可安全回滚。
def snapshot_kv_cache():
    # 优化：预分配缓存的数据不需要复制。检查点只保存 buffer 引用和有效长度，
    # 被拒绝的候选 token 会在下一轮写入同一位置，因此回滚是 O(1) 的。
    return kv_cache, kv_cache_length, kv_cache_capacity


def restore_kv_cache(cache_snapshot):
    global kv_cache, kv_cache_length, kv_cache_capacity
    kv_cache, kv_cache_length, kv_cache_capacity = cache_snapshot


def truncate_kv_cache(length):
    """将所有层的 KV Cache 回滚到 ``length`` 个已处理 token。"""
    global kv_cache_length
    if length < 0:
        raise ValueError("KV Cache length must be non-negative")
    if length > kv_cache_length:
        raise ValueError("Cannot extend a KV Cache while rolling it back")
    # 优化：无需切片或复制 K/V Tensor，只改变下一次写入的位置。
    kv_cache_length = length


def prepare_params_for_inference(params):
    """原地把 NumPy 权重包装为 Torch Tensor，避免每个 token 重复转换。"""
    def convert(value):
        if isinstance(value, dict):
            for key, child in value.items():
                value[key] = convert(child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                value[index] = convert(child)
        elif not torch.is_tensor(value):
            # torch.as_tensor 在 CPU 上与 NumPy 权重共享内存，不额外复制 1558M 参数。
            return torch.as_tensor(value)
        return value

    return convert(params)


def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    x_max = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=-1, keepdim=True)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    # 优化：推理开始时权重已转换为 Tensor；这里不再为每一层重复拷贝 g/b。
    g = torch.as_tensor(g_b['g'], dtype=x.dtype, device=x.device)
    b = torch.as_tensor(g_b['b'], dtype=x.dtype, device=x.device)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return x_hat * g + b

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    # 优化：prepare_params_for_inference 后该分支直接复用权重 Tensor。
    w = torch.as_tensor(w, dtype=x.dtype, device=x.device)
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    return x @ w + b
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    h = linear(x, w_b1)
    h = gelu(h)
    return linear(h, w_b2)


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    d_k = k.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores + mask
    weights = softmax(scores)
    return weights @ v


def mha(x, attn, n_head, layer_idx, n_past: int = 0):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # 优化：把全部 attention head 组织成一个批次 [head, seq, head_dim]。
    # 原实现按 head 循环，会让 1558M 的每层产生 25 次小矩阵运算和 25 次 cat。
    n_seq, three_n_embd = x.shape
    head_dim = three_n_embd // (3 * n_head)
    q, k_new, v_new = x.reshape(n_seq, 3, n_head, head_dim).permute(1, 2, 0, 3).unbind(0)

    if kv_cache:
        cached_k, cached_v = kv_cache[layer_idx]
        if cached_k is None:
            # 仅在第一次进入该层时分配；容量由本次生成的 prompt + 最大新 token 数给出。
            cached_k = torch.empty(
                (n_head, kv_cache_capacity, head_dim), dtype=k_new.dtype, device=k_new.device
            )
            cached_v = torch.empty(
                (n_head, kv_cache_capacity, head_dim), dtype=v_new.dtype, device=v_new.device
            )
            kv_cache[layer_idx][0] = cached_k
            kv_cache[layer_idx][1] = cached_v

        write_end = n_past + n_seq
        if write_end > kv_cache_capacity:
            raise ValueError("KV Cache capacity is smaller than the requested sequence length")
        # 优化：原来的 torch.cat 会在每次解码复制整个历史；这里仅原地写入新 token。
        cached_k[:, n_past:write_end].copy_(k_new)
        cached_v[:, n_past:write_end].copy_(v_new)
        k, v = cached_k[:, :write_end], cached_v[:, :write_end]
    else:
        k, v = k_new, v_new

    # [n_q, n_k] 的 mask 会自动广播到所有 head。
    n_q, n_k = q.shape[1], k.shape[1]
    causal_mask = torch.zeros((n_q, n_k), dtype=q.dtype, device=q.device)
    blocked = torch.triu(
        torch.ones((n_q, n_k), dtype=torch.bool, device=q.device), diagonal=1 + n_past
    )
    causal_mask.masked_fill_(blocked, float("-inf"))

    # 优化：一次 batched matmul 完成所有 head 的 attention。
    out_heads = attention(q, k, v, causal_mask)
    x = out_heads.permute(1, 0, 2).reshape(n_seq, -1)
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head, layer_idx, n_past: int = 0):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, layer_idx=layer_idx, n_past=n_past)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


@torch.inference_mode()
def gpt2(inputs, params, n_head, n_past: int = 0):  # [n_seq] -> [n_seq, n_vocab]
    global kv_cache_length
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(n_past, n_past + len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.as_tensor(x)
    # forward pass through n_layer transformer blocks
    for layer_idx, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, layer_idx=layer_idx, n_past=n_past)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    logits = x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

    # 所有 layer 都已写入本轮 token 后，再统一推进有效长度。
    if kv_cache:
        kv_cache_length = n_past + len(inputs)
    return logits


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    if n_tokens_to_generate <= 0:
        clear_kv_cache()
        return []

    # 优化：只转换一次模型参数，并按实际需要的长度预分配 KV Cache。
    prepare_params_for_inference(params)
    reset_kv_cache(
        len(params["blocks"]), n_head, max_seq_len=len(inputs) + n_tokens_to_generate
    )

    #prefill
    logits = gpt2(inputs, params, n_head, n_past=0)
    next_id = int(torch.argmax(logits[-1]).item())
    inputs.append(next_id)

    for _ in tqdm(range(1, n_tokens_to_generate), "generating"): # auto-regressive decode loop
        n_past = kv_cache_len()
        logits = gpt2([next_id], params, n_head=n_head, n_past=n_past)  # model forward pass
        next_id = int(torch.argmax(logits[-1]).item())  # greedy sampling
        inputs.append(next_id)  # append prediction to input
    
    generate_ids = inputs[-n_tokens_to_generate:]
    clear_kv_cache()
    return generate_ids  # only return generated ids

def greedy_speculative_generate(
    inputs,
    draft_params,
    target_params,
    hparams_draft,
    hparams_target,
    n_tokens_to_generate,
    K,
    fallback_after_rejections=2,
    fallback_on_first_mismatch=True,
):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).
            fallback_after_rejections (int): 连续多少轮首 token 被拒绝后切回目标模型
                自回归；设为 0 可关闭该低接受率止损策略。
            fallback_on_first_mismatch (bool): 若首个草稿 token 已与目标 token 不同，
                是否跳过整段验证并立即切回目标模型自回归。

        Returns:
            list: A list of newly generated token IDs.
            
    """
    if n_tokens_to_generate <= 0:
        clear_kv_cache()
        return []
    if K <= 0:
        raise ValueError("K must be a positive integer")
    if fallback_after_rejections < 0:
        raise ValueError("fallback_after_rejections must be non-negative")

    draft_n_head = hparams_draft["n_head"]
    target_n_head = hparams_target["n_head"]
    max_context = min(hparams_draft["n_ctx"], hparams_target["n_ctx"])
    if len(inputs) + n_tokens_to_generate >= max_context:
        raise ValueError("Prompt plus generated tokens exceeds the model context length")

    generated_ids = []
    consecutive_zero_accepts = 0

    try:
        # 优化：模型权重在整个生成请求中只做一次 NumPy -> Tensor 包装。
        prepare_params_for_inference(draft_params)
        prepare_params_for_inference(target_params)

        # 修改：分别预填充两套 cache。全局 kv_cache 是原代码的存储方式，
        # 因此在模型之间切换时用 snapshot/restore 保存各自的状态。
        cache_capacity = len(inputs) + n_tokens_to_generate
        reset_kv_cache(len(draft_params["blocks"]), draft_n_head, max_seq_len=cache_capacity)
        draft_next_logits = gpt2(inputs, draft_params, draft_n_head, n_past=0)
        draft_cache = snapshot_kv_cache()

        reset_kv_cache(len(target_params["blocks"]), target_n_head, max_seq_len=cache_capacity)
        target_next_logits = gpt2(inputs, target_params, target_n_head, n_past=0)
        target_cache = snapshot_kv_cache()

        def finish_with_target_greedy(active_target_cache, next_logits):
            """复用已有 target cache 完成剩余贪心生成，不需要再次 prefill。"""
            restore_kv_cache(active_target_cache)
            while len(generated_ids) < n_tokens_to_generate:
                next_id = int(torch.argmax(next_logits[-1]).item())
                generated_ids.append(next_id)
                if len(generated_ids) < n_tokens_to_generate:
                    next_logits = gpt2(
                        [next_id],
                        target_params,
                        target_n_head,
                        n_past=kv_cache_len(),
                    )

        # 优化：本轮第一个候选必须与已有的 target prefill logits 相同才可能被接受。
        # 若它已经不同，直接验证 K 个候选只会额外消耗一次大模型前向；复用 target
        # cache 转为普通贪心能消除这段负开销，且输出与目标模型严格相同。
        if fallback_on_first_mismatch:
            first_draft_id = int(torch.argmax(draft_next_logits[-1]).item())
            first_target_id = int(torch.argmax(target_next_logits[-1]).item())
            if first_draft_id != first_target_id:
                finish_with_target_greedy(target_cache, target_next_logits)
                return generated_ids

        while len(generated_ids) < n_tokens_to_generate:
            batch_size = min(K, n_tokens_to_generate - len(generated_ids))

            # 修改：草稿模型贪心地产生最多 K 个候选 token。产生第 i+1 个候选
            # 前，将第 i 个候选写入它自己的 KV Cache。
            restore_kv_cache(draft_cache)
            draft_prefix_len = kv_cache_len()
            proposed_ids = []
            draft_logits = draft_next_logits
            for proposal_index in range(batch_size):
                proposed_id = int(torch.argmax(draft_logits[-1]).item())
                proposed_ids.append(proposed_id)
                if proposal_index < batch_size - 1:
                    draft_logits = gpt2(
                        [proposed_id],
                        draft_params,
                        draft_n_head,
                        n_past=kv_cache_len(),
                    )
            # 此时草稿 cache 少最后一个候选 token；保留它用于发生拒绝时回滚。
            draft_after_proposals = snapshot_kv_cache()

            # 修改：目标模型一次前向传播验证整段候选。第一个候选应与此前
            # prefill 的 logits 比较，之后第 i 个候选与第 i-1 行验证 logits 比较。
            restore_kv_cache(target_cache)
            target_prefix_len = kv_cache_len()
            verification_logits = gpt2(
                proposed_ids,
                target_params,
                target_n_head,
                n_past=target_prefix_len,
            )
            target_after_proposals = snapshot_kv_cache()
            # 优化：把 K 次 Python 侧 argmax 合并为一次向量化 argmax。
            target_prediction_logits = torch.cat(
                [target_next_logits[-1:], verification_logits[:-1]], dim=0
            )
            target_ids = torch.argmax(target_prediction_logits, dim=-1).tolist()

            accepted_count = 0
            for proposed_id, target_id in zip(proposed_ids, target_ids):
                if proposed_id != target_id:
                    break
                accepted_count += 1

            generated_ids.extend(proposed_ids[:accepted_count])
            if accepted_count == 0:
                consecutive_zero_accepts += 1
            else:
                consecutive_zero_accepts = 0
            fallback_to_target = (
                fallback_after_rejections > 0
                and consecutive_zero_accepts >= fallback_after_rejections
                and len(generated_ids) < n_tokens_to_generate
            )

            if accepted_count == batch_size:
                # 修改：整段被接受。目标模型最后一行 logits 已经是下一 token
                # 的预测；草稿模型只需补写最后一个候选 token 即可继续下一轮。
                target_cache = target_after_proposals
                target_next_logits = verification_logits[-1:]

                restore_kv_cache(draft_after_proposals)
                draft_next_logits = gpt2(
                    [proposed_ids[-1]],
                    draft_params,
                    draft_n_head,
                    n_past=kv_cache_len(),
                )
                draft_cache = snapshot_kv_cache()
            else:
                # 修改：首个不一致位置直接采用目标模型的贪心 token。两套 cache
                # 都回滚到「已接受候选」后，再写入该 token；这样被拒绝的草稿
                # token 永远不会污染下一轮推理。
                correction_id = target_ids[accepted_count]
                generated_ids.append(correction_id)

                restore_kv_cache(target_after_proposals)
                truncate_kv_cache(target_prefix_len + accepted_count)
                target_next_logits = gpt2(
                    [correction_id],
                    target_params,
                    target_n_head,
                    n_past=kv_cache_len(),
                )
                target_cache = snapshot_kv_cache()

                if not fallback_to_target:
                    # 若马上切回目标模型，草稿 cache 不会再被使用，省去这次同步前向。
                    restore_kv_cache(draft_after_proposals)
                    truncate_kv_cache(draft_prefix_len + accepted_count)
                    draft_next_logits = gpt2(
                        [correction_id],
                        draft_params,
                        draft_n_head,
                        n_past=kv_cache_len(),
                    )
                    draft_cache = snapshot_kv_cache()

            # 优化：若草稿模型连续在首 token 就被拒绝，继续投机会额外付出
            # “目标模型验证 + 修正 token”两次前向的成本。此时直接复用当前目标
            # cache 切回普通贪心，既不重新 prefill，也不会改变任何生成结果。
            if fallback_to_target:
                finish_with_target_greedy(target_cache, target_next_logits)
                break

        return generated_ids
    finally:
        # 函数结束后不把任一模型的 cache 泄漏给普通 generate 调用。
        clear_kv_cache()


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "1558M", models_dir: str = "models", draft_model_size: str = "124M", speculative_k: int = 4, use_speculative: bool = True, fallback_after_rejections: int = 2, fallback_on_first_mismatch: bool = True):
    from utils import load_encoder_hparams_and_params

    # 修改：投机推理默认同时加载 124M 草稿模型与 model_size（默认 1558M）
    # 目标模型。二者共用 GPT-2 的 tokenizer，因此使用目标模型的 encoder 即可。
    encoder, hparams_target, target_params = load_encoder_hparams_and_params(model_size, models_dir)
    if use_speculative:
        _, hparams_draft, draft_params = load_encoder_hparams_and_params(draft_model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    if use_speculative:
        max_context = min(hparams_draft["n_ctx"], hparams_target["n_ctx"])
    else:
        max_context = hparams_target["n_ctx"]
    assert len(input_ids) + n_tokens_to_generate < max_context

    # generate output ids
    start = time.time()
    if use_speculative:
        output_ids = greedy_speculative_generate(
            input_ids,
            draft_params,
            target_params,
            hparams_draft,
            hparams_target,
            n_tokens_to_generate,
            speculative_k,
            fallback_after_rejections,
            fallback_on_first_mismatch,
        )
    else:
        # 保留普通自回归路径，便于与投机推理的输出/耗时进行对照。
        output_ids = generate(input_ids, target_params, hparams_target["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)
