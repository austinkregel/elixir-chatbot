defmodule Brain.Memory.EmbedderTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Memory.Embedder

  setup _context do
    ensure_pubsub_started()
    ensure_started(Embedder)

    :ok
  end

  describe "initialization" do
    test "embedder has ready?/0 status function" do
      ready = Embedder.ready?()
      assert is_boolean(ready)
    end
  end

  describe "build_vocabulary" do
    test "builds vocabulary from texts" do
      texts = ["hello world", "hello there", "goodbye world"]

      {:ok, vocab_size} = Embedder.build_vocabulary(texts)

      assert vocab_size > 0
      assert Embedder.ready?()
      assert Embedder.vocabulary_size() == vocab_size
    end

    test "filters out words with frequency < 2" do
      texts = ["hello hello hello", "world world", "unique"]

      {:ok, vocab_size} = Embedder.build_vocabulary(texts)
      assert vocab_size == 2
    end
  end

  describe "embed" do
    test "embed/1 returns error or embedding based on ready state" do
      result = Embedder.embed("hello")

      case Embedder.ready?() do
        false ->
          assert {:error, :not_ready} = result

        true ->
          assert {:ok, embedding} = result
          assert is_list(embedding)
      end
    end

    test "returns embedding vector after vocabulary is built" do
      texts = ["hello world", "hello there", "world peace"]
      {:ok, _} = Embedder.build_vocabulary(texts)

      {:ok, embedding} = Embedder.embed("hello world")

      assert is_list(embedding)
      assert embedding != []
      assert Enum.all?(embedding, &is_float/1)
    end

    test "similar texts produce similar embeddings" do
      texts = ["hello world", "hello there friend", "goodbye world", "good morning world"]

      {:ok, _} = Embedder.build_vocabulary(texts)

      {:ok, emb1} = Embedder.embed("hello world")
      {:ok, emb2} = Embedder.embed("hello there")
      {:ok, emb3} = Embedder.embed("goodbye cruel")

      sim_hello = Embedder.cosine_similarity(emb1, emb2)
      sim_different = Embedder.cosine_similarity(emb1, emb3)
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
      :ok = Embedder.load_model(model)
      assert Embedder.ready?()
      vocab_size = Embedder.vocabulary_size()
      assert vocab_size == map_size(model.vocabulary)
    end
  end
end
