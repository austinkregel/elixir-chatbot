defmodule World.MetricsTest do
  @moduledoc "Tests for World.Metrics record_document, diff, summary."
  use ExUnit.Case, async: false

  alias World.Metrics

  describe "record_document" do
    test "increments document count" do
      m = Metrics.new()
      m = Metrics.record_document(m, 10, 2, 50)
      assert m.documents_processed == 1
      assert m.total_tokens == 10
      assert m.total_sentences == 2
      m = Metrics.record_document(m, 5, 1, 25)
      assert m.documents_processed == 2
      assert m.total_tokens == 15
    end
  end

  describe "diff" do
    test "computes entity count differences between two metrics structs" do
      m1 = Metrics.new() |> Metrics.record_document(10, 2, 50)
      m1 = %{m1 | entities_discovered: 5, entities_promoted: 2}
      m2 = Metrics.new() |> Metrics.record_document(8, 1, 30)
      m2 = %{m2 | entities_discovered: 3, entities_promoted: 1}

      result = Metrics.diff(m1, m2)
      assert result.entity_count_diff == 2
      assert result.promoted_diff == 1
    end
  end

  describe "summary" do
    test "returns non-empty map" do
      m = Metrics.new() |> Metrics.record_document(20, 4, 100)
      summary = Metrics.summary(m)
      assert is_map(summary)
      assert summary.documents == 1
      assert summary.tokens == 20
      assert summary.entities_discovered >= 0
    end
  end
end
