defmodule Brain.Analysis.FeatureExtractor.SchemaContractTest do
  @moduledoc """
  The vector the *pipeline* emits must match the manifest that names it.

  `ChunkFeaturesTest`'s `dimension_manifest/0` block already proves the manifest
  is internally consistent: names are unique, `group_widths/0` accounts for
  every dimension in contiguous runs, the fingerprint covers order, and groups
  10 and 17 agree on the lexical-domain vocabulary. It also checks emission
  length — but against a hand-built `%ChunkAnalysis{text: ..., pos_tags: []}`,
  one sentence, with the POS tagger bypassed.

  That is the shape production never sees. This file closes the gap by driving
  `Pipeline.analyze_chunk/1` end to end, which is the path
  `Brain.materialize_profiles/1` and `mix gen_micro_data` both take, over
  sentences chosen to exercise different branches — a command, a question, a
  negation, an unknown proper noun, a sentence fragment.

  It also checks something no existing test does: that every emitted value is a
  finite number. A `NaN` propagates through the nearest-centroid distance
  computation and poisons every comparison silently, because `NaN != NaN` makes
  the affected class simply never win. The assertion names the group and
  dimension at the offending index rather than reporting a bare position, since
  "index 328 is NaN" is not actionable and "entity_type_semantics/etype_coverage
  is NaN" is.

  Promoted from `.claude/corpus/manifest_check.exs`, which did this as an ad-hoc
  script that nothing ran.
  """

  use Brain.Test.BrainCase, async: false

  alias Brain.Analysis.FeatureExtractor
  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Analysis.Pipeline

  # Chosen for branch coverage, not realism: an imperative with a known device,
  # an interrogative, a negation, a proper noun outside every gazetteer, a bare
  # fragment with no verb, and a past-tense clause.
  @sentences [
    "Turn on the kitchen lights",
    "What's the weather in Qxlarn?",
    "Never lock the front door",
    "I just don't understand the problem, sir.",
    "also the air conditioner",
    "I was walking home yesterday"
  ]

  describe "the pipeline's emitted vector against the manifest" do
    test "every sentence emits exactly as many values as the manifest names" do
      declared = ChunkFeatures.vector_dimension()

      for text <- @sentences do
        vector = text |> Pipeline.analyze_chunk() |> FeatureExtractor.extract_vector()

        assert length(vector) == declared,
               "#{inspect(text)} emitted #{length(vector)} values for #{declared} named " <>
                 "dimensions (difference #{length(vector) - declared}). " <>
                 "Diff ChunkFeatures.group_widths/0 against the groups in extract/2 to find " <>
                 "which one moved; a group whose width depends on runtime state (23, " <>
                 "entity_type_semantics) moves without any code change."
      end
    end

    test "every emitted value is a finite number, named by group and dimension" do
      manifest = ChunkFeatures.dimension_manifest()

      for text <- @sentences do
        vector = text |> Pipeline.analyze_chunk() |> FeatureExtractor.extract_vector()

        offenders =
          vector
          |> Enum.with_index()
          |> Enum.reject(fn {value, _index} -> is_number(value) and value == value end)
          |> Enum.map(fn {value, index} ->
            {group, name} = Enum.at(manifest, index)
            "  index #{index} (#{group}/#{name}) = #{inspect(value)}"
          end)

        assert offenders == [],
               "#{inspect(text)} emitted non-finite values:\n" <>
                 Enum.join(offenders, "\n") <>
                 "\nA NaN never compares equal to itself, so the affected class silently " <>
                 "never wins a nearest-centroid match."
      end
    end

    test "the manifest describes the same schema for every sentence" do
      # dimension_manifest/0 resolves group 23's names from TypeHierarchy at
      # call time. If that moved mid-run the vectors in one snapshot would not
      # share a schema, which is the one thing a snapshot must guarantee.
      fingerprints =
        Enum.map(@sentences, fn text ->
          _ = Pipeline.analyze_chunk(text)
          ChunkFeatures.schema_fingerprint()
        end)

      assert Enum.uniq(fingerprints) |> length() == 1,
             "the extractor schema changed during a single run: #{inspect(fingerprints)}. " <>
               "Group 23 reads TypeHierarchy.parent_types/0 live, so a graph write mid-run " <>
               "makes the vectors in one snapshot incomparable to each other."
    end
  end

  describe "run provenance" do
    test "capture!/0 records a schema fingerprint that matches the extractor" do
      provenance = Brain.Analysis.RunProvenance.capture!()

      assert provenance.extractor.schema_fingerprint == ChunkFeatures.schema_fingerprint()
      assert provenance.extractor.vector_dimension == ChunkFeatures.vector_dimension()
    end

    test "capture!/0 records the group widths, so a drift is attributable" do
      provenance = Brain.Analysis.RunProvenance.capture!()
      widths = Map.new(ChunkFeatures.group_widths(), fn {g, w} -> {to_string(g), w} end)

      assert provenance.extractor.group_widths == widths

      assert provenance.extractor.group_widths
             |> Map.values()
             |> Enum.sum() == ChunkFeatures.vector_dimension()
    end

    test "capture!/0 records the AGE graph's parent types, which set group 23's width" do
      provenance = Brain.Analysis.RunProvenance.capture!()
      parent_types = Brain.Analysis.TypeHierarchy.parent_types()

      assert provenance.age_graph.parent_type_count == length(parent_types)

      # The +2 is etype_coherence and etype_coverage. Stated as an assertion
      # rather than a comment because it is the link that makes the recorded
      # parent-type count explain a width change rather than merely accompany
      # one.
      assert provenance.extractor.group_widths["entity_type_semantics"] ==
               length(parent_types) + 2
    end
  end
end
