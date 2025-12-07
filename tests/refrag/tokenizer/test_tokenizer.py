import pytest
from refrag.model.tokenization_refrag import RefragTokenizer, from_pretrained


def test_parse_and_pack_simple():
    txt = "Hello <doc id=\"1\">This is a doc</doc> world"
    tok = from_pretrained("prajjwal1/bert-mini", "sshleifer/tiny-gpt2", compression=4)
    segs = tok.parse_and_tokenize(txt)
    # Expect three segments: decoder 'Hello ', encoder 'This is a doc', decoder ' world'
    types = [s["type"] for s in segs]
    assert types[0] == "decoder"
    assert types[1] == "encoder"
    assert types[2] == "decoder"

    enc_seg = segs[1]
    assert "packed_tokens" in enc_seg
    assert "attention_mask" in enc_seg
    # packed should be a list of rows with width == compression
    for row in enc_seg["packed_tokens"]:
        assert len(row) == 4
    for m in enc_seg["attention_mask"]:
        assert len(m) == 4


if __name__ == "__main__":
    pytest.main(["-q", "tests/test_tokenizer.py"])
