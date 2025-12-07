from refrag.model.tokenizer.tokenizer_utils import (
    compute_prefix_strings,
    compute_dp_matrix,
    alignment_pairs_from_dp,
)


def test_prefix_strings_and_dp():
    # simple token ids (here tokens are strings for clarity)
    dec_ids = ['a', 'b', 'c']
    enc_ids = ['a', 'b', 'c']

    dec_prefixes = compute_prefix_strings(lambda ids: ''.join(ids), dec_ids)
    enc_prefixes = compute_prefix_strings(lambda ids: ''.join(ids), enc_ids)

    assert dec_prefixes == ['', 'a', 'ab', 'abc']
    assert enc_prefixes == ['', 'a', 'ab', 'abc']

    dp = compute_dp_matrix(decoder_ids=dec_ids, encoder_ids=enc_ids,
                           dec_decode_fn=lambda ids: ''.join(ids),
                           enc_decode_fn=lambda ids: ''.join(ids))

    pairs = alignment_pairs_from_dp(dp)
    # expected aligned prefix lengths: (1,1), (2,2), (3,3)
    assert (1, 1) in pairs
    assert (2, 2) in pairs
    assert (3, 3) in pairs


def test_partial_alignment():
    dec_ids = ['The', ' cat', ' sat']
    enc_ids = ['The cat', ' sat']

    dp = compute_dp_matrix(decoder_ids=dec_ids, encoder_ids=enc_ids,
                           dec_decode_fn=lambda ids: ''.join(ids),
                           enc_decode_fn=lambda ids: ''.join(ids))

    pairs = alignment_pairs_from_dp(dp)
    # Expect at least one alignment when prefixes match
    assert any(isinstance(p, tuple) and len(p) == 2 for p in pairs)

