defmodule Brain.Analysis.TemporalResolverTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.TemporalResolver, as: TR

  @ref ~D[2024-06-15]

  describe "resolve/2" do
    test "relative expressions resolve against the reference date" do
      assert %{date: ~D[2024-06-15], granularity: :day} = TR.resolve("today", @ref)
      assert %{date: ~D[2024-06-14], granularity: :day} = TR.resolve("yesterday", @ref)
      assert %{date: ~D[2024-06-16], granularity: :day} = TR.resolve("tomorrow", @ref)
      assert %{date: ~D[2024-06-22], granularity: :day} = TR.resolve("next week", @ref)
    end

    test "absolute expressions carry the right granularity" do
      assert %{date: ~D[2024-01-01], granularity: :year} = TR.resolve("2024", @ref)
      assert %{date: ~D[2024-01-01], granularity: :month} = TR.resolve("in January 2024", @ref)
      assert %{date: ~D[2024-01-03], granularity: :day} = TR.resolve("January 3 2024", @ref)
    end

    test "prepositions and surrounding words are ignored" do
      assert %{granularity: :day} = TR.resolve("on Monday", @ref)
    end

    test "unresolvable text returns nil (best-effort, no crash)" do
      assert TR.resolve("bananas", @ref) == nil
      assert TR.resolve("", @ref) == nil
      assert TR.resolve(nil, @ref) == nil
    end
  end

  describe "order/3" do
    test "orders resolvable dates and reports :unknown otherwise" do
      assert TR.order("yesterday", "tomorrow", @ref) == :before
      assert TR.order("tomorrow", "yesterday", @ref) == :after
      assert TR.order("today", "today", @ref) == :equal
      assert TR.order("bananas", "today", @ref) == :unknown
    end
  end

  describe "contains?/3" do
    test "a coarser span contains a finer date within it" do
      assert TR.contains?("2024", "January 3 2024", @ref)
      assert TR.contains?("in January 2024", "January 3 2024", @ref)
    end

    test "does not contain finer-over-coarser or out-of-period" do
      refute TR.contains?("January 2024", "2024", @ref)
      refute TR.contains?("2023", "January 3 2024", @ref)
      refute TR.contains?("today", "today", @ref)
    end
  end
end
