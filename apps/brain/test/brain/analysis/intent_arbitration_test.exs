defmodule Brain.Analysis.IntentArbitrationTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.IntentArbitration

  # `n` disagreements of one pair of domains in `fold`, with how many of them
  # each route was right on.
  defp examples(fold, classifier_domain, profile_domain, n, right) do
    for i <- 1..n do
      %{
        fold: fold,
        classifier_domain: classifier_domain,
        profile_domain: profile_domain,
        right: Map.new(right, fn {route, count} -> {route, i <= count} end)
      }
    end
  end

  defp fit(examples, granularity, strength, routes \\ [:classifier, :domain, :speech_act]) do
    IntentArbitration.fit(examples, routes: routes, granularity: granularity, prior_strength: strength, language: "en")
  end

  describe "decide/3" do
    test "takes the route the pair's smoothed rate says is most often right" do
      model =
        fit(
          examples(0, "smalltalk", "communication", 20, classifier: 15, domain: 2, speech_act: 1) ++
            examples(0, "weather", "smarthome", 20, classifier: 2, domain: 14, speech_act: 1),
          :pair,
          1
        )

      assert {:route, :classifier, _} = IntentArbitration.decide(model, "smalltalk", "communication")
      assert {:route, :domain, _} = IntentArbitration.decide(model, "weather", "smarthome")
    end

    test "a pair no example fell in has the rate over every example" do
      model = fit(examples(0, "smalltalk", "communication", 10, classifier: 8, domain: 1, speech_act: 0), :pair, 5)

      probabilities = IntentArbitration.probabilities(model, "music", "code")
      assert probabilities == %{classifier: 0.8, domain: 0.1, speech_act: 0.0}
      assert {:route, :classifier, ^probabilities} = IntentArbitration.decide(model, "music", "code")
    end

    test "equally probable routes are no decision: it abstains" do
      model = fit(examples(0, "smalltalk", "music", 10, classifier: 4, domain: 4, speech_act: 1), :global, 1)

      assert {:abstain, %{classifier: same, domain: same}} = IntentArbitration.decide(model, "smalltalk", "music")
    end

    test "a model of two routes decides between those two only" do
      model = fit(examples(0, "a", "b", 10, classifier: 3, domain: 6), :global, 1, [:classifier, :domain])

      assert {:route, :domain, probabilities} = IntentArbitration.decide(model, "a", "b")
      assert Map.keys(probabilities) |> Enum.sort() == [:classifier, :domain]
    end
  end

  describe "select/2" do
    test "keeps a pair's own counts when they decide the folds better than the overall rate" do
      by_pair =
        for fold <- 0..4 do
          examples(fold, "smalltalk", "communication", 20, classifier: 18, domain: 1, speech_act: 0) ++
            examples(fold, "weather", "smarthome", 20, classifier: 1, domain: 18, speech_act: 0)
        end

      {model, report} = IntentArbitration.select(List.flatten(by_pair), routes: [:classifier, :domain, :speech_act], language: "en")

      assert model.granularity in [:pair, :classifier_domain, :profile_domain]
      assert model.selection.chosen == {model.granularity, model.prior_strength}
      assert report.folds == [0, 1, 2, 3, 4]
      assert {:route, :domain, _} = IntentArbitration.decide(model, "weather", "smarthome")
    end

    test "on a flat surface takes the coarsest granularity and the largest strength" do
      flat = for fold <- 0..4, do: examples(fold, "smalltalk", "communication", 10, classifier: 7, domain: 2, speech_act: 1)

      {model, _report} = IntentArbitration.select(List.flatten(flat), routes: [:classifier, :domain, :speech_act], language: "en")

      assert model.granularity == :global
      assert model.prior_strength == 10_000
    end

    test "fewer than two folds holding a disagreement raise" do
      assert_raise ArgumentError, ~r/one-standard-error rule needs two/, fn ->
        IntentArbitration.select(examples(0, "a", "b", 5, classifier: 2, domain: 1), routes: [:classifier, :domain], language: "en")
      end
    end
  end

  describe "examples" do
    test "an example that does not say whether a route was right raises" do
      assert_raise ArgumentError, ~r/does not say whether \[:speech_act\]/, fn ->
        fit(examples(0, "a", "b", 3, classifier: 1, domain: 1), :global, 1)
      end
    end
  end
end
