defmodule Brain.ML.CorpusManagerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.CorpusManager

  describe "corpus_size/0" do
    test "returns a non-negative integer" do
      size = CorpusManager.corpus_size()
      assert is_integer(size)
      assert size >= 0
    end
  end

  describe "size_by_category/0" do
    test "returns breakdown by category" do
      result = CorpusManager.size_by_category()

      assert is_map(result)
      assert Map.has_key?(result, :training_data)
      assert Map.has_key?(result, :ml_models)
      assert Map.has_key?(result, :evaluation)
      assert Map.has_key?(result, :training_worlds)
      assert Map.has_key?(result, :knowledge)
      assert Map.has_key?(result, :total)

      assert is_integer(result.training_data)
      assert is_integer(result.ml_models)
      assert result.total == CorpusManager.corpus_size()
    end
  end

  describe "can_add?/1" do
    test "returns true for small additions" do
      assert CorpusManager.can_add?(1024)
    end

    test "returns false for additions exceeding limit" do
      refute CorpusManager.can_add?(CorpusManager.max_size() + 1)
    end

    test "returns true for zero bytes" do
      assert CorpusManager.can_add?(0)
    end
  end

  describe "max_size/0" do
    test "returns 50 GB in bytes" do
      assert CorpusManager.max_size() == 50 * 1024 * 1024 * 1024
    end
  end

  describe "utilization_percent/0" do
    test "returns a float between 0 and 100" do
      pct = CorpusManager.utilization_percent()
      assert is_float(pct)
      assert pct >= 0.0
      assert pct <= 100.0
    end
  end

  describe "format_bytes/1" do
    test "formats bytes" do
      assert CorpusManager.format_bytes(500) == "500 B"
    end

    test "formats kilobytes" do
      assert CorpusManager.format_bytes(1536) == "1.5 KB"
    end

    test "formats megabytes" do
      assert CorpusManager.format_bytes(5 * 1024 * 1024) == "5.0 MB"
    end

    test "formats gigabytes" do
      assert CorpusManager.format_bytes(2 * 1024 * 1024 * 1024) == "2.0 GB"
    end

    test "formats zero bytes" do
      assert CorpusManager.format_bytes(0) == "0 B"
    end

    test "formats boundary values" do
      assert CorpusManager.format_bytes(1023) == "1023 B"
      assert CorpusManager.format_bytes(1024) == "1.0 KB"
      assert CorpusManager.format_bytes(1024 * 1024) == "1.0 MB"
      assert CorpusManager.format_bytes(1024 * 1024 * 1024) == "1.0 GB"
    end
  end
end
