# NSL-test
**Self-attention** Bilibili Li Hongyi’s explanation video: https://www.bilibili.com/video/BV1Xp4y1b7ih

KV Cache introduction: https://zhuanlan.zhihu.com/p/630832593

---

This is an incomplete implementation of GPT-2. 

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer.
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `NSL-gpt2.py` contains the actual GPT model and generation code which we can run as a python script, but it is an incomplete version. Believe that you can successfully complete it😎👍.

#### Dependencies
```bash
pip install -r requirements.txt
```
#### Usage

The first run requires downloading the model, which is slow, please be patient.

```bash
python NSL-gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40
```

Which generates

```
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

We used **124M** model in this test. You can also control the number of tokens to generate, the model size (one of `["124M", "355M", "774M", "1558M"]`), and the directory to save the models:

```bash
python NSL-gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "124M" \
    --models_dir "models"
```

When you write the greedy_speculative_decoding function, you need to load both the **124M and 1558M** models at the same time, and be careful to modify the parameters when loading the models.

When we use the **1558M** model for autoregressive inference, the returned results are as follows: (When using greedy sampling, this result is unique)
```
 so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T
```

### Greedy speculative decoding

`NSL-gpt2.py` also implements greedy speculative decoding. By default, the
target model is **1558M** and the draft model is **124M**. The target model
verifies a group of draft tokens in one forward pass; a rejected draft token is
replaced by the target model's greedy token. Therefore, its generated text is
identical to normal greedy decoding with the 1558M target model.

```bash
python NSL-gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "1558M" \
    --draft_model_size "124M" \
    --speculative_k 4
```

`--speculative_k` is the number of tokens proposed per round. To measure the
1558M autoregressive baseline with the same script, set
`--use_speculative=False`.

### Latency-oriented implementation details

The speculative path further reduces Python and memory-management overhead
without changing greedy decisions:

* KV tensors are preallocated once for the prompt plus requested output length.
  Appending a token is an in-place write, while rollback only restores a cache
  length; it does not concatenate or copy the historical K/V tensors.
* Attention runs all heads as one batched operation instead of invoking a
  separate attention and `torch.cat` for every head.
* If the initial draft token already differs from the target token, the decoder
  immediately reuses the target KV cache and finishes with normal target-model
  greedy decoding instead of verifying a doomed K-token block. It also applies
  the same fallback after consecutive first-token rejections in later rounds.
  This avoids negative speedups on low-acceptance prompts while keeping the
  output exactly unchanged. Set `--fallback_on_first_mismatch=False` and
  `--fallback_after_rejections=0` to disable these safeguards.

For CPU measurements, test several `--speculative_k` values (usually 2, 4,
6, and 8) on the same prompt and output length. The best value depends on the
draft acceptance rate and the target machine's matrix-multiplication behavior.
