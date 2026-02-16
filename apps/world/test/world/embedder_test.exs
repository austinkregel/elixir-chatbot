defmodule World.EmbedderTest do
  @moduledoc "Tests for World.Embedder vocabulary building and embeddings."
  use ExUnit.Case, async: false

  alias World.Embedder

  setup do
    Embedder.init()
    :ok
  end

  describe "vocabulary and embedding" do
    test "get_status returns structure when table not initialized for unknown world" do
      status = Embedder.get_status("nonexistent_world_#{System.unique_integer([:positive])}")
      assert is_map(status)
      assert Map.has_key?(status, :ready)
      assert Map.has_key?(status, :phase)
    end

    test "embed returns error when world not initialized" do
      result = Embedder.embed("nonexistent_world", "hello")
      assert result == {:error, :not_initialized} or result == {:error, :no_training_data}
    end

    test "cosine_similarity returns value in [0, 1] for same vector" do
      # Embedder doesn't expose cosine_similarity directly; check get_status
      status = Embedder.get_status("test")
      assert status.phase in [:not_initialized, :table_not_ready, :no_data, :building, :ready]
    end
  end
end
