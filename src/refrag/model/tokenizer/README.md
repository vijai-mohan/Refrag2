Tokenizer utilities

This directory contains small, dependency-free helper functions used to align encoder and decoder tokenizations.

Functions

- compute_prefix_strings(decode_fn, ids)
  - Inputs: decode_fn(ids_slice) -> str, ids: List[int or token-like]
  - Returns: list of decoded prefixes for prefix lengths 0..len(ids). The 0-length prefix is the empty string.
  - Example: if ids represent characters ['a','b','c'], and decode_fn joins them into a string, the result is ['', 'a', 'ab', 'abc'].

- compute_dp_matrix(decoder_ids, encoder_ids, dec_decode_fn, enc_decode_fn)
  - Builds a boolean DP matrix of shape (len(encoder_ids)+1) x (len(decoder_ids)+1).
  - dp[r][c] is True iff the decoded prefix of the first r encoder tokens equals the decoded prefix of the first c decoder tokens.
  - The matrix includes the empty prefix at row 0 and column 0.

- alignment_pairs_from_dp(dp)
  - Converts a DP matrix into a list of aligned prefix-length pairs (dec_idx, enc_idx), ignoring the zero-length prefixes.
  - Each returned pair (c, r) corresponds to dp[r][c] == True and r>0 and c>0.

Usage example

These utilities are intentionally small. You can use them with HF tokenizers by passing a lambda like `lambda ids: tokenizer.decode(ids)` as the decode functions.

Example:

- Tokenize text with the encoder tokenizer and the decoder tokenizer.
- Call `compute_dp_matrix(decoder_ids, encoder_ids, dec_decode_fn, enc_decode_fn)`.
- Call `alignment_pairs_from_dp(dp)` to get aligned token index pairs.

These pairs can be used to match embeddings and logits between encoder (projected) and decoder token positions when the decoded text prefixes match exactly.

Dynamic programming alignment walkthrough

Goal: compute where the encoder and decoder represent the same text prefix, even if they split the text differently into tokens.

Definitions
- Decoder tokens (columns): d0..dC-1; prefixes c = 0..C. Prefix c means the first c decoder tokens.
- Encoder tokens (rows): e0..eR-1; prefixes r = 0..R. Prefix r means the first r encoder tokens.
- dp[r][c] is True if decoder and encoder decode to the same text on those prefixes.

Building the table
1) Build all decoded prefix strings for both sides:
   - dec_prefixes[c] = dec_decode_fn(decoder_ids[:c]) for c = 0..C
   - enc_prefixes[r] = enc_decode_fn(encoder_ids[:r]) for r = 0..R
2) For every cell (r, c), set dp[r][c] = (enc_prefixes[r] == dec_prefixes[c]).

Example table (decoder columns, encoder rows)

Text: "The cat sat"

Suppose tokenizers split differently:
- Decoder tokens: ["The", " cat", " sat"] (3 tokens)
- Encoder tokens: ["The cat", " sat"] (2 tokens)

Prefixes include the empty prefix at index 0. T denotes equal decoded prefixes.

          |   0   |   1 (The)   |   2 (The cat)   |   3 (The cat sat)
----------+-------+-------------+-----------------+--------------------
 0 ("")   |   T   |      F      |        F        |         F
 1 (The cat)
          |   F   |      F      |        T        |         F
 2 (The cat sat)
          |   F   |      F      |        F        |         T

Interpretation
- dp[0][0] = T because both empty prefixes decode to empty string.
- dp[1][2] = T because encoder prefix 1 decodes to "The cat" and decoder prefix 2 also decodes to "The cat".
- dp[2][3] = T because both full prefixes decode to the full string "The cat sat".

Choosing aligned token positions
- Each True cell (r, c) with r>0 and c>0 indicates that token index (c-1) on the decoder side aligns with token index (r-1) on the encoder side.
- For the example above, aligned token indices are (c-1, r-1) ∈ {(1, 0), (2, 1)} corresponding to dp[1][2] and dp[2][3].
- In practice, you can use all True cells (simple and robust), or extract a strict monotonic path if you want a one-to-one mapping.

Algorithm steps using the helpers
1) Tokenize your text into decoder_ids and encoder_ids.
2) Compute the DP matrix:
   - `dp = compute_dp_matrix(decoder_ids, encoder_ids, dec_decode_fn, enc_decode_fn)`
3) Extract aligned pairs:
   - `pairs = alignment_pairs_from_dp(dp)`  # list of (dec_prefix_len, enc_prefix_len)
4) Convert pairs to token indices for losses:
   - token indices are (dec_idx - 1, enc_idx - 1) for each pair with dec_idx>0 and enc_idx>0.
5) Compute losses only at those aligned positions.

Complexity and tips
- Building all decoded prefixes is O(C + R) decode calls; filling the table is O(C * R) string compares.
- For typical prompt lengths (tens to a few hundred tokens), this is very fast and easy to reason about.
- Make sure decode functions skip special tokens on both sides, or strip them first, to avoid spurious mismatches.
- Be consistent about whitespace handling; using the tokenizer’s decode is safest and consistent across prefixes.
