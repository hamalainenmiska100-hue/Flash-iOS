"""
bench.py — MLX inference benchmark for Qwen3.5 models.
Loads a model via mlx-lm, generates tokens, reports metrics.

Usage:
    uv run bench.py --model mlx-community/Qwen3.5-35B-A3B-MLX-4bit --tokens 50
    uv run bench.py --model ./local-model-dir --tokens 100 --prompt "Hello"
"""

import argparse
import time
import sys

import psutil
import mlx.core as mx
import mlx.utils
import mlx_lm


def get_mem_gb():
    """Current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def fmt_time(seconds):
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="MLX inference benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--max-mem-gb", type=float, default=40.0,
                        help="Abort if RSS exceeds this (GB). Safety limit.")
    args = parser.parse_args()

    t_start = time.time()
    mem_before = get_mem_gb()
    print(f"[{fmt_time(0)}] Loading model: {args.model}")
    print(f"[{fmt_time(0)}] Memory before load: {mem_before:.1f} GB")

    # Load model and tokenizer
    model, tokenizer = mlx_lm.load(args.model)
    mx.eval(model.parameters())

    mem_after_load = get_mem_gb()
    t_loaded = time.time() - t_start
    print(f"[{fmt_time(t_loaded)}] Model loaded. Memory: {mem_after_load:.1f} GB (+{mem_after_load - mem_before:.1f} GB)")

    if mem_after_load > args.max_mem_gb:
        print(f"ABORT: Memory {mem_after_load:.1f} GB exceeds safety limit {args.max_mem_gb} GB")
        sys.exit(1)

    # Count parameters
    total_params = sum(p.size for n, p in mlx.utils.tree_flatten(model.parameters()))
    params_b = total_params / 1e9

    # Generate tokens with live reporting
    print(f"[{fmt_time(time.time() - t_start)}] Generating {args.tokens} tokens...")
    print(f"[{fmt_time(time.time() - t_start)}] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")

    prompt_tokens = tokenizer.encode(args.prompt)
    token_times = []
    generated_tokens = []
    peak_mem = mem_after_load

    t_gen_start = time.time()

    for i, response in enumerate(mlx_lm.stream_generate(
        model, tokenizer, prompt=args.prompt,
        max_tokens=args.tokens,
    )):
        t_now = time.time()

        if i == 0:
            ttft_ms = (t_now - t_gen_start) * 1000
            print(f"[{fmt_time(t_now - t_start)}] Token 1/{args.tokens}... ttft={ttft_ms:.0f}ms")
        else:
            token_times.append(t_now - t_prev)

        t_prev = t_now
        generated_tokens.append(response.token)

        # Periodic status
        if (i + 1) % 5 == 0 or i == args.tokens - 1:
            cur_mem = get_mem_gb()
            peak_mem = max(peak_mem, cur_mem)
            elapsed = t_now - t_gen_start
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[{fmt_time(t_now - t_start)}] Token {i+1}/{args.tokens}... "
                  f"{tps:.1f} tok/s (mem: {cur_mem:.1f} GB)")

            if cur_mem > args.max_mem_gb:
                print(f"ABORT: Memory {cur_mem:.1f} GB exceeds safety limit {args.max_mem_gb} GB")
                sys.exit(1)

        if i + 1 >= args.tokens:
            break

    t_gen_end = time.time()
    total_gen_time = t_gen_end - t_gen_start
    total_tokens = len(generated_tokens)
    tok_sec = total_tokens / total_gen_time if total_gen_time > 0 else 0
    peak_mem = max(peak_mem, get_mem_gb())

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)

    # Summary
    print(f"\n[{fmt_time(time.time() - t_start)}] Done. {total_tokens} tokens in {total_gen_time:.1f}s")
    print(f"Generated: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
    print()

    # Compute active params for MoE (rough heuristic: if model name contains A{N}B)
    active_b = params_b  # default: dense model
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    import re
    match = re.search(r'A(\d+)B', model_name, re.IGNORECASE)
    if match:
        active_b = float(match.group(1))

    # Machine-parseable result line
    print(f"RESULT model={model_name} params_B={params_b:.1f} active_B={active_b:.1f} "
          f"tok_sec={tok_sec:.2f} ttft_ms={ttft_ms:.0f} mem_gb={peak_mem:.1f} "
          f"tokens={total_tokens}")


if __name__ == "__main__":
    main()
