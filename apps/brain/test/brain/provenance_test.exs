defmodule Brain.ProvenanceTest do
  @moduledoc """
  The per-request trace collector behind task 039's criterion that a value from
  a fallback default must be distinguishable from a computed one.

  Two properties carry the weight. Collection must be genuinely off by default,
  or every production request pays for a feature only the harness uses. And it
  must never leave a process collecting after a failure, or one request's
  provenance silently attaches to the next.
  """
  use ExUnit.Case, async: true

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Provenance

  setup do
    # A previous test leaving the flag set would make the next one pass for the
    # wrong reason, so each starts from a known state.
    if Provenance.collecting?(), do: Provenance.stop()
    :ok
  end

  describe "off by default" do
    test "collecting?/0 is false in a fresh process" do
      refute Provenance.collecting?()
    end

    test "record/4 returns the value unchanged and stores nothing" do
      assert Provenance.record(["a"], 0.5, :default, source: "test") == 0.5
      refute Provenance.collecting?()
    end

    test "record/4 does not validate when collection is off, so it costs nothing" do
      # An undeclared origin raises only while collecting. The check is part of
      # recording, not of the no-op path, which is what keeps an instrumented
      # site cheap on an ordinary request.
      assert Provenance.record(["a"], 0.5, :not_an_origin, source: "test") == 0.5
    end
  end

  describe "collect/1" do
    test "returns the result and the entries in the order they happened" do
      {result, entries} =
        Provenance.collect(fn ->
          Provenance.record(["first"], 1, :computed, source: "test")
          Provenance.record(["second"], 2, :default, source: "test")
          :done
        end)

      assert result == :done
      assert Enum.map(entries, & &1.path) == [["first"], ["second"]]
      assert Enum.map(entries, & &1.origin) == [:computed, :default]
    end

    test "turns collection off afterwards" do
      Provenance.collect(fn -> :ok end)
      refute Provenance.collecting?()
    end

    test "turns collection off even when the function raises, and re-raises" do
      assert_raise RuntimeError, "boom", fn ->
        Provenance.collect(fn -> raise "boom" end)
      end

      refute Provenance.collecting?()
    end

    test "turns collection off even when the function exits" do
      catch_exit(Provenance.collect(fn -> exit(:timeout) end))
      refute Provenance.collecting?()
    end

    test "an entry records its value, origin, source and meta" do
      {_result, [entry]} =
        Provenance.collect(fn ->
          Provenance.record(["entity", "familiarity"], 0.5, :absent,
            source: "ChunkFeatures.entity_features/1",
            meta: %{"reason" => "no accumulated context"}
          )
        end)

      assert entry == %{
               path: ["entity", "familiarity"],
               value: 0.5,
               origin: :absent,
               source: "ChunkFeatures.entity_features/1",
               meta: %{"reason" => "no accumulated context"}
             }
    end
  end

  describe "the origin vocabulary is closed" do
    test "an undeclared origin raises while collecting" do
      message =
        try do
          Provenance.collect(fn ->
            Provenance.record(["a"], 1, :guessed, source: "test")
          end)
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "is not a declared origin"
      assert message =~ "closed"
    end

    test "origins/0 names exactly the five" do
      assert Provenance.origins() == [:computed, :declared, :default, :absent, :unavailable]
    end

    test "stand_in? is true for the three that are not about this request's input" do
      assert Provenance.stand_in?(:default)
      assert Provenance.stand_in?(:absent)
      assert Provenance.stand_in?(:unavailable)
      refute Provenance.stand_in?(:computed)

      # A declared config value is a legitimate answer but the same answer for
      # every input, so it is not a stand-in and not "your input" either.
      refute Provenance.stand_in?(:declared)
    end
  end

  describe "an entry must be traceable" do
    test "a missing source raises, because an entry with no address is useless" do
      message =
        try do
          Provenance.collect(fn -> Provenance.record(["a"], 1, :computed) end)
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "needs a :source"
    end

    test "an empty or non-string path raises" do
      assert_raise ArgumentError, fn ->
        Provenance.collect(fn -> Provenance.record([], 1, :computed, source: "test") end)
      end

      assert_raise ArgumentError, fn ->
        Provenance.collect(fn -> Provenance.record([:a], 1, :computed, source: "test") end)
      end
    end
  end

  describe "summaries" do
    test "stand_ins/1 keeps only the values not derived from the input" do
      {_r, entries} =
        Provenance.collect(fn ->
          Provenance.record(["a"], 1, :computed, source: "t")
          Provenance.record(["b"], 2, :declared, source: "t")
          Provenance.record(["c"], 3, :default, source: "t")
          Provenance.record(["d"], 4, :unavailable, source: "t")
        end)

      assert Provenance.stand_ins(entries) |> Enum.map(& &1.path) == [["c"], ["d"]]
    end

    test "census/1 counts every origin, including the ones with none" do
      {_r, entries} =
        Provenance.collect(fn ->
          Provenance.record(["a"], 1, :computed, source: "t")
          Provenance.record(["b"], 2, :computed, source: "t")
          Provenance.record(["c"], 3, :default, source: "t")
        end)

      assert Provenance.census(entries) == %{
               computed: 2,
               declared: 0,
               default: 1,
               absent: 0,
               unavailable: 0
             }
    end

    test "missing_sources/2 names the instrumented modules that reported nothing" do
      {_r, entries} =
        Provenance.collect(fn ->
          Provenance.record(["a"], 1, :computed, source: "TypeHierarchy.config/2")
        end)

      assert Provenance.missing_sources(entries, ["TypeHierarchy.", "ChunkFeatures."]) ==
               ["ChunkFeatures."]
    end
  end

  describe "a real subsystem reports through it" do
    test "TypeHierarchy.config/2 separates a declared value from a fallback" do
      {_result, entries} =
        Provenance.collect(fn ->
          TypeHierarchy.config("default_propn_type", "person")
          TypeHierarchy.config("a_key_entity_types_does_not_declare", "my fallback")
        end)

      by_path = Map.new(entries, &{&1.path, &1})

      declared = by_path[["config", "default_propn_type"]]
      assert declared.origin == :declared
      assert declared.value == "person"
      assert declared.meta["file"] == "priv/analysis/entity_types.json"

      fell_back = by_path[["config", "a_key_entity_types_does_not_declare"]]
      assert fell_back.origin == :default
      assert fell_back.value == "my fallback"
      assert fell_back.meta["reason"] =~ "declares no"
    end

    test "the returned values are unchanged by being recorded" do
      {_r, _entries} = Provenance.collect(fn -> :ok end)

      assert TypeHierarchy.config("default_propn_type", "person") ==
               Provenance.collect(fn -> TypeHierarchy.config("default_propn_type", "person") end)
               |> elem(0)
    end
  end
end
