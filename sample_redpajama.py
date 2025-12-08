import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


# -----------------------------
# CONFIG
# -----------------------------
DATASET_NAME = "togethercomputer/RedPajama-Data-1T"
# 50/50 mix of ArXiv and Books.
SUBSETS: List[Tuple[str, float]] = [("default", 1.0)]
TEXT_FIELD = "text"

TOKENIZER_NAME = "gpt2"                    # change to LLaMA, T5, etc.
TARGET_TOKENS = 10_000_000              # 10M tokens
SHARD_TOKENS = 10_000_00                  # ~1M tokens per output file

OUTPUT_DIR = "redpajama_1b_sample"
# -----------------------------


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    ensure_dir(OUTPUT_DIR)
    tokenizer = get_tokenizer(TOKENIZER_NAME)

    total_tokens = 0
    shard_tokens = 0
    shard_index = 0
    tokens_by_subset: Dict[str, int] = {}
    targets_by_subset: Dict[str, int] = {}

    weight_sum = sum(w for _, w in SUBSETS)
    for subset, weight in SUBSETS:
        targets_by_subset[subset] = int(TARGET_TOKENS * (weight / weight_sum))
        tokens_by_subset[subset] = 0

    # Prepare streaming iterators per subset
    subset_iters: Dict[str, any] = {}
    exhausted: set = set()
    for subset, _ in SUBSETS:
        print(f"Streaming subset: {subset}", flush=True)
        subset_iters[subset] = iter(load_dataset(DATASET_NAME, subset, split="train", streaming=True))

    shard_path = os.path.join(OUTPUT_DIR, f"shard_{shard_index:05d}.jsonl")
    shard_file = open(shard_path, "w", encoding="utf-8")

    try:
        while total_tokens < TARGET_TOKENS:
            all_hit_targets = True
            for subset, _ in SUBSETS:
                if subset in exhausted:
                    continue
                if tokens_by_subset[subset] >= targets_by_subset[subset]:
                    continue
                all_hit_targets = False
                try:
                    ex = next(subset_iters[subset])
                except StopIteration:
                    print(f"Subset {subset} exhausted early.")
                    exhausted.add(subset)
                    continue
                if TEXT_FIELD not in ex:
                    continue
                text = ex[TEXT_FIELD]
                n_tokens = count_tokens(tokenizer, text)

                total_tokens += n_tokens
                tokens_by_subset[subset] += n_tokens
                shard_tokens += n_tokens

                out = {"subset": subset, TEXT_FIELD: text}
                shard_file.write(json.dumps(out, ensure_ascii=False) + "\n")

                if shard_tokens >= SHARD_TOKENS:
                    shard_file.close()
                    shard_index += 1
                    shard_tokens = 0
                    shard_path = os.path.join(OUTPUT_DIR, f"shard_{shard_index:05d}.jsonl")
                    shard_file = open(shard_path, "w", encoding="utf-8")
                    print(f"New shard: {shard_path} (total_tokens={total_tokens})", flush=True)

                if total_tokens >= TARGET_TOKENS:
                    break
            # If every subset either hit target or is exhausted, stop.
            if all(
                (tokens_by_subset[s] >= targets_by_subset[s]) or (s in exhausted)
                for s, _ in SUBSETS
            ):
                break

        print(f"Done. Total tokens collected: {total_tokens}")
        for subset in tokens_by_subset:
            print(f"  {subset}: {tokens_by_subset[subset]} tokens (target {targets_by_subset[subset]})")
        print(f"Data in directory: {OUTPUT_DIR}")

    finally:
        shard_file.close()


if __name__ == "__main__":
    main()
