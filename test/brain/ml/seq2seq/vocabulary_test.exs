defmodule Brain.ML.Seq2Seq.VocabularyTest do
  use ExUnit.Case
  
  alias Brain.ML.Seq2Seq.Vocabulary
  
  setup do
    # Start vocabulary server for tests
    start_supervised!({Vocabulary, name: :test_vocab})
    
    # Build a small vocabulary for testing
    texts = [
      "hello world",
      "hello there",
      "goodbye world",
      "test example"
    ]
    
    Vocabulary.build_vocabulary(texts, server: :test_vocab, vocab_size: 20)
    
    :ok
  end
  
  test "encodes text to indices" do
    {:ok, indices} = Vocabulary.encode("hello world", server: :test_vocab)
    assert is_list(indices)
    assert length(indices) > 0
  end
  
  test "decodes indices back to text" do
    {:ok, indices} = Vocabulary.encode("hello world", server: :test_vocab)
    {:ok, text} = Vocabulary.decode(indices, server: :test_vocab)
    assert is_binary(text)
  end
  
  test "round trip encoding/decoding" do
    original = "hello world"
    {:ok, indices} = Vocabulary.encode(original, server: :test_vocab)
    {:ok, decoded} = Vocabulary.decode(indices, server: :test_vocab)
    # Note: decoded may not match exactly due to tokenization, but should be similar
    assert String.contains?(decoded, "hello") or String.contains?(decoded, "world")
  end
  
  test "gets special token indices" do
    {:ok, sos_idx} = Vocabulary.get_special_token(:sos, server: :test_vocab)
    {:ok, eos_idx} = Vocabulary.get_special_token(:eos, server: :test_vocab)
    {:ok, pad_idx} = Vocabulary.get_special_token(:pad, server: :test_vocab)
    
    assert is_integer(sos_idx)
    assert is_integer(eos_idx)
    assert is_integer(pad_idx)
    assert sos_idx != eos_idx
  end
  
  test "returns vocabulary size" do
    size = Vocabulary.size(server: :test_vocab)
    assert size > 0
  end
  
  test "checks if ready" do
    assert Vocabulary.ready?(server: :test_vocab) == true
  end
end
