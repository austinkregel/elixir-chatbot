defmodule Brain.Analysis.ChunkProfileRegressionTest do
  @moduledoc """
  Regression tests using gold_standard.json to verify ChunkProfile
  derived labels land in expected semantic neighborhoods.

  Run with: mix test --only regression
  """

  use ExUnit.Case, async: false
  @moduletag :regression
  @moduletag timeout: 300_000

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
        assert profile.polarity in [:affirmative, :negative]
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

  defp load_gold_standard do
    path =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv/evaluation/intent/gold_standard.json"
        priv -> Path.join(to_string(priv), "evaluation/intent/gold_standard.json")
      end

    path |> File.read!() |> Jason.decode!()
  end
end
