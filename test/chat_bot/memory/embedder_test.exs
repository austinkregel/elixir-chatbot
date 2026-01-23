defmodule ChatBot.Memory.EmbedderTest do
  use ExUnit.Case, async: false

  alias ChatBot.Memory.Embedder

  setup do
    # Start or restart the embedder for each test
    case Process.whereis(Embedder) do
      nil -> Embedder.start_link()
      pid -> GenServer.stop(pid) && Embedder.start_link()
    end

    :ok
  end

  describe "initialization" do
    test "starts not ready" do
      # Re-start to get fresh state
      GenServer.stop(Embedder)
      {:ok, _} = Embedder.start_link()

      refute Embedder.ready?()
      assert Embedder.vocabulary_size() == 0
    end
  end

  describe "build_vocabulary" do
    test "builds vocabulary from texts" do
      texts = [
        "hello world",
        "hello there",
        "goodbye world"
      ]

      {:ok, vocab_size} = Embedder.build_vocabulary(texts)

      assert vocab_size > 0
      assert Embedder.ready?()
      assert Embedder.vocabulary_size() == vocab_size
    end

    test "filters out words with frequency < 2" do
      texts = [
        "hello hello hello",
        "world world",
        "unique"
      ]

      {:ok, vocab_size} = Embedder.build_vocabulary(texts)

      # "unique" appears only once, should be filtered out
      # "hello" and "world" should remain
      assert vocab_size == 2
    end
  end

  describe "embed" do
    test "returns error when not ready" do
      GenServer.stop(Embedder)
      {:ok, _} = Embedder.start_link()

      assert {:error, :not_ready} = Embedder.embed("hello")
    end

    test "returns embedding vector after vocabulary is built" do
      texts = ["hello world", "hello there", "world peace"]
      {:ok, _} = Embedder.build_vocabulary(texts)

      {:ok, embedding} = Embedder.embed("hello world")

      assert is_list(embedding)
      assert length(embedding) > 0
      assert Enum.all?(embedding, &is_float/1)
    end

    test "similar texts produce similar embeddings" do
      texts = [
        "hello world",
        "hello there friend",
        "goodbye world",
        "good morning world"
      ]

      {:ok, _} = Embedder.build_vocabulary(texts)

      {:ok, emb1} = Embedder.embed("hello world")
      {:ok, emb2} = Embedder.embed("hello there")
      {:ok, emb3} = Embedder.embed("goodbye cruel")

      sim_hello = Embedder.cosine_similarity(emb1, emb2)
      sim_different = Embedder.cosine_similarity(emb1, emb3)

      # "hello world" should be more similar to "hello there" than to "goodbye cruel"
      assert sim_hello > sim_different
    end
  end

  describe "cosine_similarity" do
    test "returns 1.0 for identical vectors" do
      vec = [0.5, 0.5, 0.5]
      assert_in_delta Embedder.cosine_similarity(vec, vec), 1.0, 0.0001
    end

    test "returns 0.0 for orthogonal vectors" do
      a = [1.0, 0.0]
      b = [0.0, 1.0]
      assert_in_delta Embedder.cosine_similarity(a, b), 0.0, 0.0001
    end
  end

  describe "model export/load" do
    test "exports and loads model" do
      texts = ["hello world", "hello there", "world peace"]
      {:ok, _} = Embedder.build_vocabulary(texts)

      {:ok, model} = Embedder.export_model()

      assert is_map(model)
      assert Map.has_key?(model, :vocabulary)
      assert Map.has_key?(model, :idf_weights)

      # Stop and restart, then load
      GenServer.stop(Embedder)
      {:ok, _} = Embedder.start_link()
      refute Embedder.ready?()

      :ok = Embedder.load_model(model)
      assert Embedder.ready?()
    end
  end
end
