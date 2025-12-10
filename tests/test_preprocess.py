
from src.pipeline.preprocess import clean_text, split_sentences

def test_clean_text_basic():
    assert clean_text("  Hello   world  ") == "Hello world"

def test_split_sentences():
    sents = split_sentences("Hi! Test. OK?")
    assert sents == ["Hi!", "Test.", "OK?"]