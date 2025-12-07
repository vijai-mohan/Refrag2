import pytest
from transformers import AutoTokenizer
from refrag.model.tokenization_refrag import RefragTokenizer

@pytest.fixture(scope="module")
def encoder_tokenizer():
    return AutoTokenizer.from_pretrained("prajjwal1/bert-mini")

@pytest.fixture(scope="module")
def decoder_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

def test_simple_parsing(encoder_tokenizer, decoder_tokenizer):
    """Tests parsing of a simple string with one doc tag."""
    refrag_tok = RefragTokenizer(encoder_tokenizer, decoder_tokenizer, compression=4)
    text = "This is a test. <doc id='1'>This is a document.</doc>"
    segments = refrag_tok.parse_segments(text)
    
    assert len(segments) == 2
    assert segments[0]['type'] == 'decoder'
    assert segments[0]['raw_text'] == "This is a test. "
    assert segments[1]['type'] == 'encoder'
    assert segments[1]['raw_text'] == "This is a document."
    assert segments[1]['meta']['id'] == '1'

def test_multiple_doc_tags(encoder_tokenizer, decoder_tokenizer):
    """Tests parsing of a string with multiple doc tags."""
    refrag_tok = RefragTokenizer(encoder_tokenizer, decoder_tokenizer, compression=4)
    text = "First part. <doc id='1'>Doc 1.</doc> Second part. <doc id='2'>Doc 2.</doc>"
    segments = refrag_tok.parse_segments(text)
    
    assert len(segments) == 4
    assert segments[0]['type'] == 'decoder'
    assert segments[1]['type'] == 'encoder'
    assert segments[2]['type'] == 'decoder'
    assert segments[3]['type'] == 'encoder'
    assert segments[1]['meta']['id'] == '1'
    assert segments[3]['meta']['id'] == '2'

def test_tokenization_and_packing(encoder_tokenizer, decoder_tokenizer):
    """Tests that tokenization and packing work as expected."""
    refrag_tok = RefragTokenizer(encoder_tokenizer, decoder_tokenizer, compression=2)
    text = "<doc id='1'>one two three four</doc>"
    segments = refrag_tok.parse_and_tokenize(text)
    
    assert len(segments) == 1
    encoder_segment = segments[0]
    assert encoder_segment['type'] == 'encoder'
    
    # prajjwal1/bert-mini tokenizes 'one', 'two', 'three', 'four' to single tokens
    # Let's verify the packed output
    packed_tokens = encoder_segment['packed_tokens']
    assert len(packed_tokens) == 2 # 4 tokens, compression=2 -> 2 rows
    assert len(packed_tokens[0]) == 2
    assert len(packed_tokens[1]) == 2

def test_no_doc_tags(encoder_tokenizer, decoder_tokenizer):
    """Tests parsing of a string with no doc tags."""
    refrag_tok = RefragTokenizer(encoder_tokenizer, decoder_tokenizer, compression=4)
    text = "This is a string with no doc tags."
    segments = refrag_tok.parse_segments(text)
    
    assert len(segments) == 1
    assert segments[0]['type'] == 'decoder'
    assert segments[0]['raw_text'] == text

def test_empty_string(encoder_tokenizer, decoder_tokenizer):
    """Tests parsing of an empty string."""
    refrag_tok = RefragTokenizer(encoder_tokenizer, decoder_tokenizer, compression=4)
    text = ""
    segments = refrag_tok.parse_segments(text)
    assert len(segments) == 0
