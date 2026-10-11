import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

kv_cache = []

def reset_kv_cache(n_layer, n_head):
    global kv_cache
    kv_cache = [[[None, None] for _ in range(n_head)] for _ in range(n_layer)]

def clear_kv_cache():
    global kv_cache
    kv_cache = []

def kv_cache_len():
    if not kv_cache or kv_cache[0][0][0] is None:
        return 0
    return kv_cache[0][0][0].shape[0]


# 投机推理会在草稿模型和目标模型之间切换。下面两个函数只复制 cache 的
# ``list`` 结构，而不复制其中很大的 K/V Tensor；mha 每次追加 token 时都会
# 生成新的 Tensor，因此这个浅拷贝足以保存一个可回滚的 cache 检查点。
def snapshot_kv_cache():
    return [[head[:] for head in layer] for layer in kv_cache]


def restore_kv_cache(cache_snapshot):
    global kv_cache
    # 再复制一次 list，避免下一次 mha 写 cache 时修改传入的检查点。
    kv_cache = [[head[:] for head in layer] for layer in cache_snapshot]


def truncate_kv_cache(length):
    """将所有层的 KV Cache 回滚到 ``length`` 个已处理 token。"""
    if length < 0:
        raise ValueError("KV Cache length must be non-negative")

    for layer in kv_cache:
        for head in layer:
            k, v = head
            if k is None:
                continue
            if length > k.shape[0]:
                raise ValueError("Cannot extend a KV Cache while rolling it back")
            # K/V 的第 0 维是序列长度；切片不会复制大 Tensor。
            head[0] = k[:length]
            head[1] = v[:length]


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
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
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
    w = torch.as_tensor(w, dtype=x.dtype)
    b = torch.as_tensor(b, dtype=x.dtype)
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
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    qkv = torch.chunk(x, 3, dim=-1)# need to modify

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    if kv_cache:
        heads_with_cache = []
        for h , (q,k,v) in enumerate(qkv_heads):
            k_old, v_old = kv_cache[layer_idx][h]
            if k_old is not None:
                k = torch.cat([k_old, k], dim=0)
                v = torch.cat([v_old, v], dim=0)
            kv_cache[layer_idx][h][0] = k
            kv_cache[layer_idx][h][1] = v
            heads_with_cache.append((q, k, v))
        qkv_heads = heads_with_cache

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    n_q = qkv_heads[0][0].shape[0]
    n_k = qkv_heads[0][1].shape[0]
    causal_mask = torch.triu(torch.full((n_q, n_k), float("-inf")), diagonal=1 + n_past) # need to modify

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1) # need to modify
    
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


def gpt2(inputs, params, n_head, n_past: int = 0):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(n_past, n_past + len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for layer_idx, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, layer_idx=layer_idx, n_past=n_past)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    if n_tokens_to_generate <= 0:
        clear_kv_cache()
        return []

    reset_kv_cache(len(params["blocks"]), n_head)

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

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    if n_tokens_to_generate <= 0:
        clear_kv_cache()
        return []
    if K <= 0:
        raise ValueError("K must be a positive integer")

    draft_n_head = hparams_draft["n_head"]
    target_n_head = hparams_target["n_head"]
    max_context = min(hparams_draft["n_ctx"], hparams_target["n_ctx"])
    if len(inputs) + n_tokens_to_generate >= max_context:
        raise ValueError("Prompt plus generated tokens exceeds the model context length")

    generated_ids = []

    try:
        # 修改：分别预填充两套 cache。全局 kv_cache 是原代码的存储方式，
        # 因此在模型之间切换时用 snapshot/restore 保存各自的状态。
        reset_kv_cache(len(draft_params["blocks"]), draft_n_head)
        draft_next_logits = gpt2(inputs, draft_params, draft_n_head, n_past=0)
        draft_cache = snapshot_kv_cache()

        reset_kv_cache(len(target_params["blocks"]), target_n_head)
        target_next_logits = gpt2(inputs, target_params, target_n_head, n_past=0)
        target_cache = snapshot_kv_cache()

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
            target_ids = [int(torch.argmax(target_next_logits[-1]).item())]
            target_ids.extend(
                int(torch.argmax(verification_logits[index - 1]).item())
                for index in range(1, batch_size)
            )

            accepted_count = 0
            for proposed_id, target_id in zip(proposed_ids, target_ids):
                if proposed_id != target_id:
                    break
                accepted_count += 1

            generated_ids.extend(proposed_ids[:accepted_count])

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

                restore_kv_cache(draft_after_proposals)
                truncate_kv_cache(draft_prefix_len + accepted_count)
                draft_next_logits = gpt2(
                    [correction_id],
                    draft_params,
                    draft_n_head,
                    n_past=kv_cache_len(),
                )
                draft_cache = snapshot_kv_cache()

        return generated_ids
    finally:
        # 函数结束后不把任一模型的 cache 泄漏给普通 generate 调用。
        clear_kv_cache()


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "1558M", models_dir: str = "models", draft_model_size: str = "124M", speculative_k: int = 4, use_speculative: bool = True):
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
