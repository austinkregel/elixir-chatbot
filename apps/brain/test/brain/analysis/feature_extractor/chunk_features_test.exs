defmodule Brain.Analysis.FeatureExtractor.ChunkFeaturesTest do
  @moduledoc """
  Failing-first regression tests for the three concentric bugs reported
  against `Brain.Analysis.FeatureExtractor.ChunkFeatures`:

    1. Bracket access on structs (analysis[:text], acc[:novelty_score],
       s[:filled_slots], entity[:type]) raises at runtime because structs
       do not implement the Access behaviour.

    2. The rescue clause inside `extract_tokens/1` re-attempts the same
       broken bracket read, so it cannot recover.

    3. `memory_context_features/1` reads `:novelty_score`,
       `:similar_episode_count`, `:graph_known`, `:repetition_score`,
       `:conversation_centroid_distance`, and `:context_length` from a
       `%Brain.Analysis.ContextAccumulator{}` struct, but none of those
       fields exist on that struct.

  These tests pass plain structs (no maps/keyword lists) into the public
  API, exactly the way `Brain.materialize_profiles/1` does in production,
  and assert that the resulting feature vector is the right shape and
  that the memory + slot dimensions reflect the input data instead of
  collapsing to zero.

  The tests intentionally pre-supply an empty `word_features` list so
  the ETS-backed `Brain.Lexicon` is never queried — these cases exercise
  the chunk-level feature aggregation only.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.{ChunkAnalysis, ContextAccumulator, SlotResult}
  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Analysis.FeatureExtractor.EnrichmentFeatures

  @memory_dims 6
  @slot_dims 6

  describe "extract/2 with a real %ChunkAnalysis{} struct" do
    test "does not raise when reading struct fields (Bug 1: bracket access on struct)" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "what time is it",
        pos_tags: []
      }

      vector =
        try do
          ChunkFeatures.extract(analysis, [])
        rescue
          e ->
            flunk(
              "ChunkFeatures.extract/2 raised on a real %ChunkAnalysis{}: " <>
                Exception.message(e)
            )
        end

      assert is_list(vector)
      assert length(vector) == ChunkFeatures.vector_dimension()
      assert Enum.all?(vector, &is_number/1)
    end

    test "extract_tokens reads the struct's :text field (Bug 2: dead rescue clause)" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "hello world from elixir today",
        pos_tags: []
      }

      vector = ChunkFeatures.extract(analysis, [])

      [token_count_norm | _] = vector

      assert token_count_norm > 0.0,
             "expected a non-zero token-count feature for non-empty :text, got #{inspect(token_count_norm)}. " <>
               "This means extract_tokens fell into its rescue branch and read an empty string."
    end
  end

  describe "memory_context_features/1 with a real %ContextAccumulator{}" do
    test "does not raise on bracket access against the struct (Bug 1)" do
      acc = %ContextAccumulator{}

      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "hi",
        pos_tags: [],
        accumulated_context: acc
      }

      assert is_list(ChunkFeatures.extract(analysis, []))
    end

    test "reflects real ContextAccumulator fields, not the nonexistent ones (Bug 3)" do
      acc = %ContextAccumulator{
        signals: [
          {:speech_act, :directive, 0.9},
          {:discourse, :user, 0.8}
        ],
        combined_confidence: 0.85,
        conflict_measure: 0.0,
        entity_familiarity: 0.95,
        relevant_episodes: [%{}, %{}, %{}, %{}, %{}],
        relevant_semantics: [%{}, %{}],
        conversation_topics: [:greetings, :time]
      }

      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "hi",
        pos_tags: [],
        accumulated_context: acc
      }

      vector = ChunkFeatures.extract(analysis, [])
      memory_features = memory_slice(vector)

      refute Enum.all?(memory_features, &(&1 == 0.0)),
             "memory_context_features collapsed to all-zeros despite a populated ContextAccumulator " <>
               "(real fields: combined_confidence=0.85, entity_familiarity=0.95, " <>
               "relevant_episodes=5, relevant_semantics=2, conversation_topics=2, signals=2). " <>
               "Got: #{inspect(memory_features)}"
    end

    test "an empty ContextAccumulator yields different memory features than a populated one" do
      empty_acc = %ContextAccumulator{}

      full_acc = %ContextAccumulator{
        signals: [{:a, 1, 0.9}, {:b, 2, 0.9}],
        combined_confidence: 0.95,
        entity_familiarity: 0.99,
        relevant_episodes: List.duplicate(%{}, 8),
        conversation_topics: [:x, :y, :z]
      }

      empty_vec =
        ChunkFeatures.extract(
          %ChunkAnalysis{chunk_index: 0, text: "hi", pos_tags: [], accumulated_context: empty_acc},
          []
        )

      full_vec =
        ChunkFeatures.extract(
          %ChunkAnalysis{chunk_index: 0, text: "hi", pos_tags: [], accumulated_context: full_acc},
          []
        )

      assert memory_slice(empty_vec) != memory_slice(full_vec),
             "memory features were identical for empty vs populated ContextAccumulator — " <>
               "the extractor is not actually reading any real fields"
    end
  end

  describe "slot_completeness_features/1 with a real %SlotResult{} struct" do
    test "does not raise on bracket access against the struct (Bug 1)" do
      slots =
        SlotResult.new("test_schema")
        |> SlotResult.fill_slot("foo", "bar", :explicit, 1.0)
        |> SlotResult.fill_slot("baz", "qux", :explicit, 1.0)

      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "hi",
        pos_tags: [],
        slots: slots
      }

      assert is_list(ChunkFeatures.extract(analysis, []))
    end

    test "filled-slot dimension reflects how many slots are filled" do
      empty = SlotResult.new("test_schema")

      filled =
        SlotResult.new("test_schema")
        |> SlotResult.fill_slot("foo", "bar", :explicit, 1.0)
        |> SlotResult.fill_slot("baz", "qux", :explicit, 1.0)
        |> SlotResult.fill_slot("zip", "zap", :explicit, 1.0)

      empty_vec =
        ChunkFeatures.extract(
          %ChunkAnalysis{chunk_index: 0, text: "hi", pos_tags: [], slots: empty},
          []
        )

      filled_vec =
        ChunkFeatures.extract(
          %ChunkAnalysis{chunk_index: 0, text: "hi", pos_tags: [], slots: filled},
          []
        )

      assert slot_slice(empty_vec) != slot_slice(filled_vec),
             "slot_completeness_features did not change between an empty and a filled %SlotResult{} — " <>
               "the extractor likely raised on bracket access and silently produced zeros"
    end
  end

  describe "dimension_manifest/0 — the vector's declared shape" do
    test "names exactly as many dimensions as vector_dimension/0 declares" do
      manifest = ChunkFeatures.dimension_manifest()

      assert length(manifest) == ChunkFeatures.vector_dimension(),
             "dimension_manifest/0 names #{length(manifest)} dimensions but " <>
               "vector_dimension/0 declares #{ChunkFeatures.vector_dimension()}. " <>
               "These cannot disagree — the width is derived from the manifest, " <>
               "so a mismatch means one of them stopped being derived."
    end

    test "every dimension name is unique across the whole vector" do
      names = Enum.map(ChunkFeatures.dimension_manifest(), fn {_g, n} -> n end)
      dupes = names -- Enum.uniq(names)

      assert dupes == [],
             "duplicate dimension names: #{inspect(Enum.uniq(dupes), limit: :infinity)}. " <>
               "A name that labels two slots makes the vector unattributable — the " <>
               "whole point of the manifest — so a new group must not reuse a prefix."
    end

    test "group_widths/0 accounts for every dimension, with no empty group" do
      widths = ChunkFeatures.group_widths()

      assert Enum.sum(Enum.map(widths, &elem(&1, 1))) == ChunkFeatures.vector_dimension()

      empty = Enum.filter(widths, fn {_g, w} -> w == 0 end)

      assert empty == [],
             "these groups contribute no dimensions: #{inspect(empty)}. " <>
               "A zero-width group is either dead code or a data file that " <>
               "failed to load — both should be loud, not a silent gap."
    end

    test "each group appears as one contiguous run, matching emission order" do
      groups = ChunkFeatures.dimension_manifest() |> Enum.map(&elem(&1, 0))
      runs = groups |> Enum.chunk_by(& &1) |> Enum.map(&hd/1)

      assert runs == Enum.uniq(runs),
             "a group's dimensions are split across the vector: #{inspect(runs)}. " <>
               "Slice helpers locate a group by its first index and width, so a " <>
               "non-contiguous group would silently read another group's values."
    end

    test "the emitted vector is exactly as long as the manifest" do
      analysis = %ChunkAnalysis{chunk_index: 0, text: "turn on the kitchen lights", pos_tags: []}
      vector = ChunkFeatures.extract(analysis, [])

      assert length(vector) == length(ChunkFeatures.dimension_manifest()),
             "extract/2 emitted #{length(vector)} values for " <>
               "#{length(ChunkFeatures.dimension_manifest())} named dimensions " <>
               "(difference #{length(vector) - length(ChunkFeatures.dimension_manifest())}). " <>
               "Compare ChunkFeatures.group_widths/0 against the groups in extract/2 " <>
               "to find which one moved."
    end

    test "schema_fingerprint/0 is deterministic and covers order, not just names" do
      assert ChunkFeatures.schema_fingerprint() == ChunkFeatures.schema_fingerprint()
      assert ChunkFeatures.schema_fingerprint() =~ ~r/^[0-9a-f]{16}$/

      # A reordering with identical names and widths must still change the
      # digest, otherwise a snapshot taken before a reorder would compare as
      # equal to one taken after it.
      manifest = ChunkFeatures.dimension_manifest()

      digest = fn m ->
        m
        |> Enum.map_join("\n", fn {g, n} -> "#{g}/#{n}" end)
        |> then(&:crypto.hash(:sha256, &1))
        |> Base.encode16(case: :lower)
        |> binary_part(0, 16)
      end

      assert digest.(manifest) == ChunkFeatures.schema_fingerprint()
      refute digest.(Enum.reverse(manifest)) == ChunkFeatures.schema_fingerprint()
    end

    test "group 10 and group 17 agree on the lexical-domain vocabulary" do
      runtime = length(Brain.Lexicon.domain_atoms())
      compiled = EnrichmentFeatures.lexicon_domain_count()

      assert runtime == compiled,
             "group 10 reads #{runtime} lexical domains from Lexicon.domain_atoms/0 at " <>
               "runtime, but EnrichmentFeatures froze #{compiled} at compile time for the " <>
               "group 17 supersenses. The vector would be built from two different " <>
               "vocabularies. Recompile brain, or reconcile the WordNet data."
    end

    test "the supersense partitions cover every lexical domain" do
      partitioned =
        EnrichmentFeatures.verb_supersense_dimension() +
          EnrichmentFeatures.noun_supersense_dimension() +
          EnrichmentFeatures.adj_adv_supersense_dimension()

      assert partitioned == EnrichmentFeatures.lexicon_domain_count(),
             "the verb_/noun_/adj_/adv_ partitions cover #{partitioned} of " <>
               "#{EnrichmentFeatures.lexicon_domain_count()} lexical domains. The " <>
               "#{EnrichmentFeatures.lexicon_domain_count() - partitioned} unmatched domain(s) " <>
               "are counted by group 10 but absent from group 17 entirely."
    end
  end

  # Locates a group's window by name rather than by arithmetic on the vector's
  # tail. The previous version subtracted a hand-maintained sum of every
  # enrichment group's width to find the memory/slot windows, so adding a group
  # anywhere but the very end silently moved these slices onto the wrong values.
  defp group_slice(vector, group) do
    manifest = ChunkFeatures.dimension_manifest()
    offset = Enum.find_index(manifest, fn {g, _n} -> g == group end)
    width = Enum.count(manifest, fn {g, _n} -> g == group end)

    refute is_nil(offset), "no group #{inspect(group)} in the dimension manifest"

    Enum.slice(vector, offset, width)
  end

  defp memory_slice(vector) do
    slice = group_slice(vector, :memory_context)
    assert length(slice) == @memory_dims
    slice
  end

  defp slot_slice(vector) do
    slice = group_slice(vector, :slot_completeness)
    assert length(slice) == @slot_dims
    slice
  end
end
