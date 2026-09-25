defmodule Brain.Analysis.ChunkProfileRegressionTest do
  @moduledoc """
  Regression tests using gold_standard.json to verify ChunkProfile
  derived labels land in expected semantic neighborhoods.

  Run with: mix test --only regression
  """

  use ExUnit.Case, async: false
  @moduletag :regression
  @moduletag timeout: 300_000

  # The pipeline reads the knowledge graph; without a checked-out connection
  # every lookup fails silently and the graph features read as "unknown".
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end

  alias Brain.Analysis.{ChunkProfile, FeatureExtractor, Pipeline}

  @intent_to_domain %{
    "account" => :account,
    "alarm" => :reminder,
    "calendar" => :calendar,
    "code" => :code,
    "communication" => :communication,
    "knowledge" => :knowledge,
    "meta" => :meta,
    "music" => :music,
    "navigation" => :navigation,
    "news" => :knowledge,
    "payment" => :account,
    "reminder" => :reminder,
    "search" => :knowledge,
    "smalltalk" => :smalltalk,
    "smarthome" => :smarthome,
    "statement" => :smalltalk,
    "timer" => :reminder,
    "todo" => :calendar,
    "weather" => :weather,
    "date" => :time,
    "time" => :time,
    "dialog" => :smalltalk,
    "display" => :smarthome,
    "status" => :meta,
    "web" => :knowledge,
    "analysis" => :knowledge
  }

  setup_all do
    Brain.TestHelpers.require_services!(:ml_inference)

    gold_standard = load_gold_standard()
    sampled = Enum.take_random(gold_standard, 100)
    {:ok, entries: sampled}
  end

  describe "gold_standard.json regression" do
    @tag :regression
    test "derived domain is in expected neighborhood for sampled entries", %{entries: entries} do
      results =
        Enum.map(entries, fn entry ->
          expected_domain_key = entry["intent"] |> String.split(".") |> List.first()
          expected_domain = Map.get(@intent_to_domain, expected_domain_key, :unknown)

          {analysis, feature_vector} = analyze_entry(entry)
          profile = ChunkProfile.materialize(analysis, feature_vector)

          %{
            text: entry["text"],
            expected_domain: expected_domain,
            actual_domain: profile.domain,
            match: profile.domain == expected_domain
          }
        end)

      match_count = Enum.count(results, & &1.match)
      total = length(results)
      accuracy = match_count / max(total, 1) * 100

      IO.puts("\nDomain accuracy: #{Float.round(accuracy, 1)}% (#{match_count}/#{total})")

      mismatches = Enum.reject(results, & &1.match)

      if length(mismatches) > 0 do
        IO.puts("Mismatches (first 10):")

        mismatches
        |> Enum.take(10)
        |> Enum.each(fn m ->
          IO.puts("  \"#{m.text}\" -> expected #{m.expected_domain}, got #{m.actual_domain}")
        end)
      end

      assert match_count / total >= 0.05,
             "Domain accuracy too low: #{Float.round(accuracy, 1)}% (need >= 5%)"
    end

    @tag :regression
    test "all profiles have valid primary axes", %{entries: entries} do
      Enum.each(Enum.take(entries, 50), fn entry ->
        {analysis, feature_vector} = analyze_entry(entry)
        profile = ChunkProfile.materialize(analysis, feature_vector)

        assert profile.speech_act_category in [
                 :assertive,
                 :directive,
                 :commissive,
                 :expressive,
                 :declarative,
                 :unknown
               ]

        assert profile.modality in [:declarative, :interrogative, :imperative, :exclamatory]

        # Polarity is a continuous negation strength, not a two-valued atom:
        # the axis manifest declares {:polarity, :continuous, {:range, 0.0, 1.0}}
        # (chunk_profile.ex), and derive_polarity/2 returns 0.0 for an
        # affirmative clause, 1.0 for a sentential negator, and a smaller
        # accumulating score for morphological negators.
        assert is_float(profile.polarity),
               "polarity should be a float in 0.0..1.0, got #{inspect(profile.polarity)}"

        assert profile.polarity >= 0.0 and profile.polarity <= 1.0

        assert profile.tense in [:past, :present, :future, :atemporal]
        assert profile.aspect in [:simple, :progressive, :perfect, :perfect_progressive]
        assert profile.urgency in [:low, :normal, :high, :critical]
        assert profile.certainty in [:committed, :tentative, :hedged, :speculative]

        assert profile.response_posture in [:direct, :hedged, :clarify, :tentative_confirm]

        assert profile.engagement_level in [
                 :passive_observation,
                 :casual_engagement,
                 :active_request,
                 :urgent_demand
               ]

        assert is_binary(profile.derived_label)
        assert is_float(profile.confidence) or is_integer(profile.confidence)
      end)
    end

    @tag :regression
    test "interaction axes are internally consistent", %{entries: entries} do
      Enum.each(Enum.take(entries, 30), fn entry ->
        {analysis, feature_vector} = analyze_entry(entry)
        profile = ChunkProfile.materialize(analysis, feature_vector)

        if profile.tense == :atemporal do
          assert profile.temporal_framing == :timeless
        end

        if profile.engagement_level == :urgent_demand do
          assert profile.urgency in [:high, :critical]
        end
      end)
    end
  end

  # -- Helpers ---------------------------------------------------------------

  # Runs `text` through the side-effect-free single-chunk analyzer and
  # returns `{ChunkAnalysis.t, feature_vector :: list(float())}`.
  #
  # `ChunkProfile.materialize/2` requires a populated feature vector to
  # exercise the feature-vector axis classifiers (`:intent_domain`,
  # `:tense_class`, etc.); passing `[]` would force every classifier to
  # reject on dimensionality and fall back to its default, which would
  # make this regression vacuously fail.
  defp analyze_entry(%{"text" => text}) when is_binary(text) do
    analysis = Pipeline.analyze_chunk(text)
    {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)
    {analysis, feature_vector}
  end

  # Training rows only. Read the whole corpus here and this regression test
  # would be profiling rows the models are evaluated on.
  defp load_gold_standard do
    Brain.ML.EvaluationStore.load_gold_standard("intent", :train)
  end
end
