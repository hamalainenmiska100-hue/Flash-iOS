"""
stream_infer.py — Streaming inference engine for Qwen3.5 MoE models.
Loads weights layer-by-layer from safetensors during inference.
Proves flash offloading works and measures overhead vs fully-loaded baseline.

Usage:
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode baseline
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode stream
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode layerwise
"""

import argparse
import time
import sys
import json
import re
import os
from pathlib import Path
from collections import defaultdict

import psutil
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import mlx_lm
from safetensors import safe_open


def get_mem_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def resolve_model_path(model_id):
    """Resolve a HF model ID to a local path."""
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def build_weight_index(model_path):
    """Build a mapping: layer_num -> [(tensor_name, file_path)].
    Also returns 'global' key for non-layer tensors (embed, norm, lm_head)."""
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)

    layer_weights = defaultdict(list)
    for name, filename in idx["weight_map"].items():
        filepath = model_path / filename
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_num = int(match.group(1))
            layer_weights[layer_num].append((name, str(filepath)))
        else:
            layer_weights["global"].append((name, str(filepath)))

    return layer_weights


def load_layer_from_safetensors(weight_index, layer_num, model, file_cache=None):
    """Load a single layer's weights from safetensors into the model.
    Uses mx.load() which handles bfloat16 natively via mmap.
    Returns the load time in seconds."""
    entries = weight_index.get(layer_num, [])
    if not entries:
        return 0.0

    t0 = time.time()

    # Group by file for efficiency
    by_file = defaultdict(list)
    for name, filepath in entries:
        by_file[filepath].append(name)

    weights = []
    for filepath, names in by_file.items():
        # mx.load() returns lazy mmap'd arrays — actual I/O at mx.eval()
        if file_cache is not None and filepath in file_cache:
            all_tensors = file_cache[filepath]
        else:
            all_tensors = mx.load(filepath)
            if file_cache is not None:
                file_cache[filepath] = all_tensors

        for name in names:
            if name not in all_tensors:
                continue
            # Sanitize: remove "language_model." prefix to match model's param paths
            san_name = name
            if san_name.startswith("language_model."):
                san_name = san_name[len("language_model."):]
            weights.append((san_name, all_tensors[name]))

    # Apply to model (language_model level)
    model.language_model.load_weights(weights, strict=False)
    # Force evaluation so weights are actually loaded from disk
    mx.eval(model.language_model.model.layers[layer_num].parameters())

    return time.time() - t0


def manual_forward(model, input_ids, cache):
    """Run the model's forward pass layer-by-layer, returning logits.
    This replicates the model's own forward but gives us per-layer control."""
    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers

    # Embed
    h = text_model.embed_tokens(input_ids)

    # Create masks (same logic as the model)
    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

    # Process layers
    for i, (layer, c) in enumerate(zip(layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)

    # Norm
    h = text_model.norm(h)

    # LM head
    if lm.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)

    return logits


def manual_forward_layerwise(model, input_ids, cache, weight_index=None, file_cache=None):
    """Same as manual_forward but returns per-layer timing info.
    If weight_index is provided, reloads weights from safetensors per layer."""
    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers

    h = text_model.embed_tokens(input_ids)
    mx.eval(h)

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

    layer_timings = []

    for i, (layer, c) in enumerate(zip(layers, cache)):
        load_time = 0.0
        if weight_index is not None:
            load_time = load_layer_from_safetensors(weight_index, i, model, file_cache)

        mask = ssm_mask if layer.is_linear else fa_mask

        t_compute = time.time()
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        compute_time = time.time() - t_compute

        layer_timings.append({
            "layer": i,
            "is_linear": layer.is_linear,
            "load_ms": load_time * 1000,
            "compute_ms": compute_time * 1000,
        })

    h = text_model.norm(h)

    if lm.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    mx.eval(logits)

    return logits, layer_timings


def generate_baseline(model, tokenizer, prompt, max_tokens):
    """Generate using mlx_lm's built-in stream_generate. Reference baseline."""
    t_start = time.time()
    token_times = []
    generated_tokens = []
    peak_mem = get_mem_gb()

    t_gen_start = time.time()

    for i, response in enumerate(mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
    )):
        t_now = time.time()

        if i == 0:
            ttft_ms = (t_now - t_gen_start) * 1000
            print(f"  [{fmt_time(t_now - t_start)}] Token 1/{max_tokens}... ttft={ttft_ms:.0f}ms")
        else:
            token_times.append(t_now - t_prev)

        t_prev = t_now
        generated_tokens.append(response.token)

        if (i + 1) % 5 == 0 or i == max_tokens - 1:
            cur_mem = get_mem_gb()
            peak_mem = max(peak_mem, cur_mem)
            elapsed = t_now - t_gen_start
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{fmt_time(t_now - t_start)}] Token {i+1}/{max_tokens}... "
                  f"{tps:.1f} tok/s (mem: {cur_mem:.1f} GB)")

        if i + 1 >= max_tokens:
            break

    total_time = time.time() - t_gen_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": ttft_ms,
        "peak_mem_gb": peak_mem,
    }


def generate_manual(model, tokenizer, prompt, max_tokens, weight_index=None, mode="stream"):
    """Generate tokens using manual layer-by-layer forward pass.
    If weight_index is provided (mode=stream), reloads weights from safetensors each layer."""
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()
    # Cache file handles for mx.load() — avoids re-parsing safetensors headers
    file_cache = {} if mode == "stream" else None

    for token_idx in range(max_tokens):
        t_token_start = time.time()

        if mode == "layerwise" or mode == "stream":
            logits, layer_timings = manual_forward_layerwise(
                model, input_ids, cache,
                weight_index=weight_index if mode == "stream" else None,
                file_cache=file_cache,
            )
            all_layer_timings.append(layer_timings)
        else:
            logits = manual_forward(model, input_ids, cache)
            layer_timings = None

        # Greedy sample
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Progress
        if token_idx == 0:
            ttft_ms = token_time * 1000
            load_ms = sum(lt["load_ms"] for lt in layer_timings) if layer_timings else 0
            compute_ms = sum(lt["compute_ms"] for lt in layer_timings) if layer_timings else 0
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (load={load_ms:.0f}ms compute={compute_ms:.0f}ms)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            load_ms = sum(lt["load_ms"] for lt in layer_timings) if layer_timings else 0
            compute_ms = sum(lt["compute_ms"] for lt in layer_timings) if layer_timings else 0
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.1f} tok/s (load={load_ms:.0f}ms compute={compute_ms:.0f}ms "
                  f"mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]  # skip prompt token
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])

        # Per-layer breakdown
        num_layers = len(gen_timings[0])
        per_layer_load = [np.mean([tt[i]["load_ms"] for tt in gen_timings]) for i in range(num_layers)]
        per_layer_compute = [np.mean([tt[i]["compute_ms"] for tt in gen_timings]) for i in range(num_layers)]
    else:
        avg_load = 0
        avg_compute = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming inference engine")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tokens", type=int, default=20, help="Tokens to generate")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--mode", choices=["baseline", "layerwise", "stream", "lazy"],
                        default="stream",
                        help="baseline=mlx_lm native, layerwise=manual forward with timing, "
                             "stream=reload weights from safetensors per layer, "
                             "lazy=load with lazy=True (mmap, OS handles paging)")
    parser.add_argument("--max-mem-gb", type=float, default=40.0,
                        help="Abort if RSS exceeds this (GB)")
    args = parser.parse_args()

    t_start = time.time()
    mem_before = get_mem_gb()

    print(f"[{fmt_time(0)}] Mode: {args.mode}")
    print(f"[{fmt_time(0)}] Loading model: {args.model}")
    print(f"[{fmt_time(0)}] Memory before load: {mem_before:.1f} GB")

    # Resolve model path (for safetensors access)
    model_path = resolve_model_path(args.model)

    if args.mode == "lazy":
        # Lazy mode: mmap weights, only pin essentials. For models > DRAM.
        model, tokenizer = mlx_lm.load(str(model_path), lazy=True)
        # Pin only embed_tokens, norm, and lm_head in DRAM
        lm = model.language_model
        text_model = lm.model
        mx.eval(text_model.embed_tokens.parameters())
        mx.eval(text_model.norm.parameters())
        if hasattr(lm, 'lm_head'):
            mx.eval(lm.lm_head.parameters())
        # Cap wired memory to leave room for the rest of the system
        wired_gb = min(args.max_mem_gb * 0.6, 28)  # ~60% of limit or 28GB max
        mx.set_wired_limit(int(wired_gb * 1024**3))
        print(f"[{fmt_time(time.time() - t_start)}] Lazy loaded. Pinned essentials, "
              f"wired limit={wired_gb:.0f}GB")
    else:
        # Full load: everything in DRAM
        model, tokenizer = mlx_lm.load(str(model_path))
        mx.eval(model.parameters())

    mem_after_load = get_mem_gb()
    t_loaded = time.time() - t_start
    print(f"[{fmt_time(t_loaded)}] Model loaded. Memory: {mem_after_load:.1f} GB "
          f"(+{mem_after_load - mem_before:.1f} GB)")

    if args.mode != "lazy" and mem_after_load > args.max_mem_gb:
        print(f"ABORT: Memory {mem_after_load:.1f} GB exceeds limit {args.max_mem_gb} GB")
        sys.exit(1)

    # Count parameters
    total_params = sum(p.size for n, p in mlx.utils.tree_flatten(model.parameters()))
    params_b = total_params / 1e9

    # Build weight index for streaming mode
    weight_index = build_weight_index(model_path) if args.mode == "stream" else None

    # Generate
    print(f"[{fmt_time(time.time() - t_start)}] Generating {args.tokens} tokens ({args.mode})...")
    print(f"[{fmt_time(time.time() - t_start)}] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")

    if args.mode == "baseline":
        result = generate_baseline(model, tokenizer, args.prompt, args.tokens)
    elif args.mode == "lazy":
        # Lazy mode uses layerwise forward (no explicit weight reload)
        # The OS handles paging mmap'd weights in/out as needed
        result = generate_manual(model, tokenizer, args.prompt, args.tokens,
                                 weight_index=None, mode="layerwise")
    else:
        result = generate_manual(model, tokenizer, args.prompt, args.tokens,
                                 weight_index=weight_index, mode=args.mode)

    # Summary
    peak_mem = max(result["peak_mem_gb"], get_mem_gb())
    print(f"\n[{fmt_time(time.time() - t_start)}] Done. {result['tokens']} tokens in "
          f"{result['total_time']:.1f}s")
    print(f"Generated: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
    print()

    # Mode-specific stats
    if args.mode in ("layerwise", "stream", "lazy"):
        print(f"Per-token breakdown (generation phase, excluding prompt):")
        print(f"  Avg weight load: {result.get('avg_load_ms_per_token', 0):.1f}ms")
        print(f"  Avg compute:     {result.get('avg_compute_ms_per_token', 0):.1f}ms")

        if result.get("per_layer_compute_ms"):
            linear_times = []
            fa_times = []
            layers = model.language_model.model.layers
            for i, t in enumerate(result["per_layer_compute_ms"]):
                if layers[i].is_linear:
                    linear_times.append(t)
                else:
                    fa_times.append(t)

            if linear_times:
                print(f"  Avg linear_attn layer: {np.mean(linear_times):.2f}ms "
                      f"({len(linear_times)} layers)")
            if fa_times:
                print(f"  Avg full_attn layer:   {np.mean(fa_times):.2f}ms "
                      f"({len(fa_times)} layers)")

        if result.get("per_layer_load_ms"):
            avg_load_per_layer = np.mean(result["per_layer_load_ms"])
            total_load = sum(result["per_layer_load_ms"])
            print(f"  Avg safetensors load/layer: {avg_load_per_layer:.2f}ms")
            print(f"  Total load per token: {total_load:.0f}ms")
        print()

    # Compute active params (MoE heuristic)
    active_b = params_b
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    match = re.search(r'A(\d+)B', model_name, re.IGNORECASE)
    if match:
        active_b = float(match.group(1))

    # Machine-parseable result line
    print(f"RESULT model={model_name} params_B={params_b:.1f} active_B={active_b:.1f} "
          f"tok_sec={result['tok_sec']:.2f} ttft_ms={result['ttft_ms']:.0f} "
          f"mem_gb={peak_mem:.1f} tokens={result['tokens']} mode={args.mode}")


if __name__ == "__main__":
    main()
