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

  # Total dims appended *after* slot_completeness by the
  # EnrichmentFeatures groups (Tier 1: 15-19; Tier 2: 20-22). Slice
  # helpers below subtract this tail so they keep pointing at the
  # memory/slot windows even as new enrichment dims land at the end
  # of the vector.
  defp enrichment_tail_dims do
    EnrichmentFeatures.wh_dimension() +
      EnrichmentFeatures.time_typology_dimension() +
      EnrichmentFeatures.verb_supersense_dimension() +
      EnrichmentFeatures.noun_supersense_dimension() +
      EnrichmentFeatures.adj_adv_supersense_dimension() +
      EnrichmentFeatures.conceptnet_edge_dimension() +
      EnrichmentFeatures.selectional_preferences_dimension() +
      EnrichmentFeatures.subcategorization_frame_dimension() +
      EnrichmentFeatures.discourse_markers_dimension() +
      EnrichmentFeatures.speech_act_wh_interaction_dimension()
  end

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

  defp memory_slice(vector) do
    offset = ChunkFeatures.vector_dimension() - enrichment_tail_dims() - @slot_dims - @memory_dims
    Enum.slice(vector, offset, @memory_dims)
  end

  defp slot_slice(vector) do
    offset = ChunkFeatures.vector_dimension() - enrichment_tail_dims() - @slot_dims
    Enum.slice(vector, offset, @slot_dims)
  end
end
