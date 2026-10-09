defmodule Brain.Analysis.ChunkProfileNounInvarianceTest do
  @moduledoc """
  Regression tests that lock in the invariant: ChunkProfile axis values
  MUST NOT change based on proper-noun token identity.

  ## The failure this test guards against

  Under the original bag-of-words `:intent_domain` micro-classifier:

  - "My name is Austin Kregel." → domain :calendar
    (because "Austin" co-occurred heavily with calendar.* training
    examples like "schedule a meeting with Austin")
  - "I live in Owosso Michigan." → domain :weather
    (because US-state tokens dominate weather.* training examples
    like "what's the weather in Michigan?")

  Neither is semantically defensible: the user's name or location should
  have zero bearing on which conversational domain they're in.

  ## The invariant

  Under the feature-vector axis classifier, token identity cannot reach
  the classifier. Therefore, for any template of the form
  "<STEM><FILLER>", the classifier's output must be identical across
  every choice of FILLER (modulo the small surface-lexical differences
  like word length, which are captured in the feature vector but are
  not expected to flip a classification decision).

  ## Contract

  There is a separate contract test asserting that the trained
  `:intent_domain` model's declared input dimensionality matches
  `ChunkFeatures.vector_dimension/0`. If the feature extractor ever
  adds or removes a dimension without a retrain, this test breaks
  instead of the classifier silently returning wrong labels.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.ChunkProfile
  alias Brain.Analysis.FeatureExtractor
  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Analysis.Pipeline
  alias Brain.ML.MicroClassifiers

  @moduletag :requires_micro_models

  # The pipeline reads the knowledge graph; without a checked-out connection
  # every lookup fails and the graph-derived features read as "unknown",
  # which is one of the inputs this file's invariant depends on.
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end

  describe "contract: axis classifier input dim matches ChunkFeatures.vector_dimension/0" do
    test ":intent_domain model's input_dim equals ChunkFeatures.vector_dimension/0" do
      case MicroClassifiers.input_dim(:intent_domain) do
        {:ok, dim} ->
          assert dim == ChunkFeatures.vector_dimension(),
                 "drift between ChunkFeatures.vector_dimension/0 " <>
                   "(#{ChunkFeatures.vector_dimension()}) and the trained " <>
                   ":intent_domain model (#{dim}). Regenerate training data " <>
                   "and retrain: `mix gen_micro_data --world default && mix train_micro`."

        {:error, :not_loaded} ->
          flunk(
            "MicroClassifiers server not started. Start the Brain app or " <>
              "exclude :requires_micro_models for this test run."
          )

        {:error, :not_trained} ->
          flunk(
            "No trained :intent_domain model found on disk. " <>
              "Run `mix gen_micro_data --world default && mix train_micro` before this test."
          )
      end
    end

    test "every axis classifier declared in ChunkProfile has a feature-vector model" do
      axes = [:intent_domain, :tense_class, :aspect_class, :urgency, :certainty_level]

      for axis <- axes do
        case MicroClassifiers.kind(axis) do
          {:ok, :feature_vector} ->
            :ok

          {:ok, other} ->
            flunk(
              "Axis classifier #{inspect(axis)} is #{inspect(other)}; " <>
                "axis classifiers must be :feature_vector to be noun-identity-immune."
            )

          {:error, reason} ->
            flunk(
              "Axis classifier #{inspect(axis)} could not be inspected: #{inspect(reason)}"
            )
        end
      end
    end
  end

  describe "invariant: proper-noun identity does not change ChunkProfile.domain" do
    test "'My name is <PERSON>.' yields the same :domain for every person" do
      names = ["Austin Kregel", "Jane Smith", "Ravi Patel", "Yuki Nakamura"]
      profiles = profiles_for_template("My name is ", ".", names)
      assert_computed(profiles, :domain, names)
      domains = Enum.map(profiles, & &1.domain)

      assert length(Enum.uniq(domains)) == 1,
             "domain swung across name permutations. pairs: " <>
               inspect(Enum.zip(names, domains))
    end

    test "'I live in <LOCATION>.' yields the same :domain for every location" do
      places = ["Owosso Michigan", "Tokyo Japan", "Madrid Spain", "Accra Ghana"]
      profiles = profiles_for_template("I live in ", ".", places)
      assert_computed(profiles, :domain, places)
      domains = Enum.map(profiles, & &1.domain)

      assert length(Enum.uniq(domains)) == 1,
             "domain swung across location permutations. pairs: " <>
               inspect(Enum.zip(places, domains))
    end

    test "'Can you help me understand <TOPIC>?' yields the same :domain for every domain-neutral topic" do
      # IMPORTANT: this test specifically locks in *identity invariance*,
      # i.e. that the model's :domain decision does not depend on the
      # specific surface tokens of the topic. It is NOT meant to assert
      # that "domain" is constant across topics whose content carries
      # legitimate domain semantics — e.g. "functional programming" is
      # genuinely a code-domain noun phrase via WordNet/ConceptNet, and
      # the model picking up :code there is correct semantic recognition,
      # not bias.
      #
      # We therefore use four domain-neutral filler phrases (deictic /
      # generic referents whose content words are abstract or
      # function-word-heavy). Any differences across these MUST be
      # noun-identity bias, the exact failure mode this test exists to
      # guard against.
      topics = [
        "the situation",
        "this idea",
        "that thing",
        "the matter"
      ]

      profiles = profiles_for_template("Can you help me understand ", "?", topics)
      assert_computed(profiles, :domain, topics)
      domains = Enum.map(profiles, & &1.domain)

      assert length(Enum.uniq(domains)) == 1,
             "domain swung across domain-neutral topic permutations. pairs: " <>
               inspect(Enum.zip(topics, domains))
    end
  end

  describe "invariant: proper-noun identity does not change other axes" do
    test "tense/aspect/urgency/certainty are stable across name permutations" do
      names = ["Austin Kregel", "Jane Smith", "Ravi Patel", "Yuki Nakamura"]

      profiles = profiles_for_template("My name is ", ".", names)

      for axis <- [:tense, :aspect, :urgency, :certainty], do: assert_computed(profiles, axis, names)

      assert_axis_stable(profiles, :tense, names)
      assert_axis_stable(profiles, :aspect, names)
      assert_axis_stable(profiles, :urgency, names)
      assert_axis_stable(profiles, :certainty, names)
    end
  end

  defp profiles_for_template(prefix, suffix, fillers) when is_list(fillers) do
    Enum.map(fillers, fn filler ->
      text = prefix <> filler <> suffix
      model = Pipeline.process(text, world_id: "default")

      [chunk | _] = model.analyses
      {feature_vector, _word_feats} = FeatureExtractor.extract(chunk)
      ChunkProfile.materialize(chunk, feature_vector)
    end)
  end

  defp assert_axis_stable(profiles, axis, fillers) do
    values = Enum.map(profiles, &Map.get(&1, axis))

    assert length(Enum.uniq(values)) == 1,
           "axis #{inspect(axis)} swung across permutations. pairs: " <>
             inspect(Enum.zip(fillers, values))
  end

  # Guards against a false-green: a stability check passes vacuously when no
  # classifier answered and every permutation holds the axis default. Asks the
  # profile's provenance whether the axis was computed, rather than whether the
  # value differs from the default -- a computed axis can legitimately land on
  # its default ("My name is X." is present tense), which the value cannot tell
  # apart from a classifier that never ran.
  defp assert_computed(profiles, axis, fillers) do
    not_computed =
      for {profile, filler} <- Enum.zip(profiles, fillers),
          not ChunkProfile.computed?(profile, axis),
          do: {filler, ChunkProfile.default_reason(profile, axis)}

    assert not_computed == [],
           "axis #{inspect(axis)} was not computed (it holds #{inspect(ChunkProfile.axis_default(axis))}) " <>
             "for: #{inspect(not_computed)}. Promote the trained classifiers with " <>
             "`MIX_ENV=test mix test.models` and rerun."
  end
end
