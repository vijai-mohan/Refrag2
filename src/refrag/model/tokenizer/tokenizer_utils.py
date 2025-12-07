"""Tokenizer helpers for encoder/decoder tokenizers and DP alignment.

Provides:
- compute_prefix_strings: returns decoded prefixes for all prefix lengths
- compute_dp_matrix: builds DP boolean matrix where dp[r][c] is True if enc_prefix(r)==dec_prefix(c)
- alignment_pairs_from_dp: returns list of (dec_idx, enc_idx) aligned pairs where both > 0

This module is intentionally small and dependency-free.
"""
from typing import List, Tuple, Callable, Any


def compute_prefix_strings(decode_fn: Callable[[List[Any]], str], ids: List[Any]) -> List[str]:
    """Return list of decoded prefixes for prefix lengths 0..len(ids).

    prefix_strings[k] == decode_fn(ids[:k])
    """
    prefixes = [""]
    for k in range(1, len(ids) + 1):
        prefixes.append(decode_fn(ids[:k]))
    return prefixes


def compute_dp_matrix(decoder_ids: List[Any], encoder_ids: List[Any],
                      dec_decode_fn: Callable[[List[Any]], str],
                      enc_decode_fn: Callable[[List[Any]], str]) -> List[List[bool]]:
    """Compute DP matrix of size (len(encoder_ids)+1) x (len(decoder_ids)+1).

    dp[r][c] corresponds to encoder prefix length r and decoder prefix length c.
    dp[r][c] == True iff enc_decode_fn(encoder_ids[:r]) == dec_decode_fn(decoder_ids[:c])
    """
    dec_prefixes = compute_prefix_strings(dec_decode_fn, decoder_ids)
    enc_prefixes = compute_prefix_strings(enc_decode_fn, encoder_ids)

    dp: List[List[bool]] = [[False] * (len(dec_prefixes)) for _ in range(len(enc_prefixes))]
    for r in range(len(enc_prefixes)):
        for c in range(len(dec_prefixes)):
            dp[r][c] = (enc_prefixes[r] == dec_prefixes[c])
    return dp


def alignment_pairs_from_dp(dp: List[List[bool]]) -> List[Tuple[int, int]]:
    """Return list of (dec_idx, enc_idx) pairs where dp[enc_idx][dec_idx] is True and both > 0."""
    pairs: List[Tuple[int, int]] = []
    if not dp:
        return pairs
    rows = len(dp)
    cols = len(dp[0])
    for r in range(1, rows):
        for c in range(1, cols):
            if dp[r][c]:
                pairs.append((c, r))
    return pairs
