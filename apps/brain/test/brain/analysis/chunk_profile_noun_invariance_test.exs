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

  describe "contract: axis classifier input dim matches ChunkFeatures.vector_dimension/0" do
    test ":intent_domain model's input_dim equals ChunkFeatures.vector_dimension/0" do
      case MicroClassifiers.input_dim(:intent_domain) do
        {:ok, dim} ->
          assert dim == ChunkFeatures.vector_dimension(),
                 "drift between ChunkFeatures.vector_dimension/0 " <>
                   "(#{ChunkFeatures.vector_dimension()}) and the trained " <>
                   ":intent_domain model (#{dim}). Regenerate training data " <>
                   "and retrain: `mix gen_micro_data && mix train_micro`."

        {:error, :not_loaded} ->
          flunk(
            "MicroClassifiers server not started. Start the Brain app or " <>
              "exclude :requires_micro_models for this test run."
          )

        {:error, :not_trained} ->
          flunk(
            "No trained :intent_domain model found on disk. " <>
              "Run `mix gen_micro_data && mix train_micro` before this test."
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
      domains = domains_for_template("My name is ", ".", names)

      refute_all_default(domains, :unknown, :domain)

      assert length(Enum.uniq(domains)) == 1,
             "domain swung across name permutations. pairs: " <>
               inspect(Enum.zip(names, domains))
    end

    test "'I live in <LOCATION>.' yields the same :domain for every location" do
      places = ["Owosso Michigan", "Tokyo Japan", "Madrid Spain", "Accra Ghana"]
      domains = domains_for_template("I live in ", ".", places)

      refute_all_default(domains, :unknown, :domain)

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

      domains = domains_for_template("Can you help me understand ", "?", topics)

      refute_all_default(domains, :unknown, :domain)

      assert length(Enum.uniq(domains)) == 1,
             "domain swung across domain-neutral topic permutations. pairs: " <>
               inspect(Enum.zip(topics, domains))
    end
  end

  describe "invariant: proper-noun identity does not change other axes" do
    test "tense/aspect/urgency/certainty are stable across name permutations" do
      names = ["Austin Kregel", "Jane Smith", "Ravi Patel", "Yuki Nakamura"]

      profiles = profiles_for_template("My name is ", ".", names)

      # Guard against false-green: at least one axis must return a non-default
      # on at least one permutation. Otherwise the classifier isn't engaging.
      nondefault_observed? =
        [
          {:tense, :present},
          {:aspect, :simple},
          {:urgency, :low},
          {:certainty, :committed}
        ]
        |> Enum.any?(fn {axis, default} ->
          profiles |> Enum.map(&Map.get(&1, axis)) |> Enum.any?(&(&1 != default))
        end)

      assert nondefault_observed?,
             "Every axis returned its default for every permutation. " <>
               "No trained axis classifier is engaging. " <>
               "Run `mix gen_micro_data && mix train_micro` and retry."

      assert_axis_stable(profiles, :tense, names)
      assert_axis_stable(profiles, :aspect, names)
      assert_axis_stable(profiles, :urgency, names)
      assert_axis_stable(profiles, :certainty, names)
    end
  end

  defp domains_for_template(prefix, suffix, fillers) do
    prefix
    |> profiles_for_template(suffix, fillers)
    |> Enum.map(& &1.domain)
  end

  defp profiles_for_template(prefix, suffix, fillers) when is_list(fillers) do
    Enum.map(fillers, fn filler ->
      text = prefix <> filler <> suffix
      model = Pipeline.process(text)

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

  # Guards against a false-green: if no trained classifier is loaded, every
  # permutation returns the axis default, and the uniqueness check passes
  # vacuously. Force the test to fail loudly in that case.
  defp refute_all_default(values, default, axis) do
    refute Enum.all?(values, &(&1 == default)),
           "axis #{inspect(axis)} returned its default (#{inspect(default)}) for " <>
             "every permutation. This usually means no trained model is loaded. " <>
             "Run `mix gen_micro_data && mix train_micro` with the new " <>
             "feature-vector classifier, then rerun this test."
  end
end
