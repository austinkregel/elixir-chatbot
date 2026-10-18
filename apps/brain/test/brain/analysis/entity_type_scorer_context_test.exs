defmodule Brain.Analysis.EntityTypeScorerContextTest do
  @moduledoc """
  How the scorer weighs a reading by its context: by the context's learned
  likelihood of the reading relative to its likelihood across every utterance
  (`Brain.Analysis.SlotTypePreferences.reading_likelihoods/2`), for entity
  types the schema lists or not and for the ordinary word alike.

  The preferences are fitted here on small fixtures and passed in, and the
  usage counts are given, so these test the scorer's arithmetic and not the
  lexicon or the gold standards. Needs the type hierarchy running
  (`Mix.Tasks.TrainSlotTypePreferences.start!/0`).
  """
  use ExUnit.Case, async: true

  alias Brain.Analysis.{EntityTypeScorer, SlotDetector, SlotTypePreferences}
  alias Brain.Analysis.EntityTypeScorer.Expected

  # Gold labels are the runtime types themselves, with no parents, so only
  # the fixtures decide the likelihoods.
  defmodule Hierarchy do
    def all_types, do: ~w(location sys_date music_artist)
    def parents_of(_type), do: []
  end

  @alpha Application.compile_env!(:brain, [:entity_type_scoring, :prior_smoothing])

  @smoothing %{
    global: 1,
    domain: 1,
    intent: 1,
    hierarchy: 0.0,
    tag_global: 1,
    tag_type: 1,
    word_global: 1,
    word_type: 1,
    numeric_global: 1,
    numeric_type: 1,
    ordinary_global: 1,
    ordinary_domain: 1,
    ordinary_intent: 1
  }

  defp example(key, intent, types, gazetteer) do
    %{
      key: key,
      intent: intent,
      spans: Enum.map(types, &%{type: &1, preceding_tag: "VERB", preceding_word: "play", numeric: false}),
      ordinary: ["NOUN"],
      gazetteer: gazetteer
    }
  end

  # Music requests name artists and their gazetteer matches are entities;
  # alarms name dates; weather requests name places, and two of every three
  # gazetteer matches in them are ordinary words.
  defp model(weather_gazetteer \\ [:ordinary, :ordinary, :entity]) do
    examples =
      for(i <- 1..6, do: example({:music, i}, "music.play", ["music_artist"], [:entity])) ++
        for(i <- 1..6, do: example({:alarm, i}, "alarm.set", ["sys_date"], [:entity])) ++
        for i <- 1..6, do: example({:weather, i}, "weather.query", ["location"], weather_gazetteer)

    SlotTypePreferences.fit(examples,
      language: "en",
      canonical: %{},
      hierarchy: Hierarchy,
      listed: SlotTypePreferences.listed_types(),
      reading_types: ~w(location sys_date music_artist),
      tag_vocabulary: ["NOUN", "VERB", :sentence_start],
      smoothing: @smoothing
    )
  end

  @candidates [%{entity_type: "sys_date", value: "Thursday"}, %{entity_type: "music_artist", value: "Thursday"}]

  # Used as a date and as a band equally often, and as an ordinary noun.
  @usage %{types: %{"sys_date" => 5, "music_artist" => 5}, ordinary: %{"noun" => 10}}

  # Starting the sentence, with no tags, so casing and part of speech say
  # nothing and only the prior and the context weigh the readings.
  defp score(expected, preferences) do
    EntityTypeScorer.score(
      @candidates,
      %{match: "Thursday", sentence_initial: true, expected_types: expected},
      usage: @usage,
      preferences: preferences
    )
  end

  defp p_entity(result, type), do: result.posteriors[type] * 1.0

  describe "expected_types/2 and domain_expected_types/2" do
    test "carry their context, the listed types' weights, and the context's reading likelihoods" do
      preferences = model()
      expected = EntityTypeScorer.expected_types("music.play", preferences: preferences)

      listed = "music.play" |> SlotDetector.get_entity_types_for_intent() |> MapSet.to_list()

      assert %Expected{context: {:intent, "music.play"}} = expected
      assert Expected.types(expected) == Enum.sort(listed)
      assert Expected.weights(expected) == SlotTypePreferences.weights(preferences, {:intent, "music.play"}, listed)
      assert expected.readings == SlotTypePreferences.reading_likelihoods(preferences, {:intent, "music.play"})

      domain = EntityTypeScorer.domain_expected_types(:weather, preferences: preferences)
      assert domain.context == {:domain, "weather"}
      assert Expected.types(domain) == :weather |> SlotDetector.entity_types_for_domain() |> MapSet.to_list() |> Enum.sort()
    end

    test "no intent is no context" do
      assert EntityTypeScorer.expected_types(nil, preferences: model()) == nil
      assert EntityTypeScorer.expected_types("", preferences: model()) == nil
    end
  end

  describe "each reading is weighed by the context's learned likelihood of it" do
    test "the entity readings' odds move by exactly the ratio of their likelihoods" do
      preferences = model()
      neutral = score(nil, preferences)

      for intent <- ["music.play", "alarm.set", "weather.query"] do
        expected = EntityTypeScorer.expected_types(intent, preferences: preferences)
        result = score(expected, preferences)

        ratio = Expected.reading_likelihood(expected, "music_artist") / Expected.reading_likelihood(expected, "sys_date")

        assert_in_delta p_entity(result, "music_artist") / p_entity(result, "sys_date"),
                        ratio * p_entity(neutral, "music_artist") / p_entity(neutral, "sys_date"),
                        1.0e-9,
                        "under #{intent}"
      end
    end

    test "music reads Thursday as the band and an alarm as the date, from the data alone" do
      preferences = model()

      assert score(EntityTypeScorer.expected_types("music.play", preferences: preferences), preferences).entity_type ==
               "music_artist"

      assert score(EntityTypeScorer.expected_types("alarm.set", preferences: preferences), preferences).entity_type ==
               "sys_date"
    end

    test "the ordinary word is weighed by the context's ordinary-word rate over the overall rate" do
      preferences = model()
      expected = EntityTypeScorer.expected_types("weather.query", preferences: preferences)
      result = score(expected, preferences)

      # weight(ordinary) / weight(type) = uses x K(o) / ((count + alpha) x K(type)).
      expected_odds =
        10 * Expected.reading_likelihood(expected, :ordinary) /
          ((5 + @alpha) * Expected.reading_likelihood(expected, "sys_date"))

      assert_in_delta result.p_ordinary / p_entity(result, "sys_date"), expected_odds, 1.0e-9
    end

    test "where the gazetteer's matches are ordinary words more often, the ordinary reading gains" do
      preferences = model()
      neutral = score(nil, preferences).p_ordinary

      weather = score(EntityTypeScorer.expected_types("weather.query", preferences: preferences), preferences).p_ordinary
      music = score(EntityTypeScorer.expected_types("music.play", preferences: preferences), preferences).p_ordinary

      assert weather > neutral
      assert music < neutral
    end

    test "the ordinary reading's weight follows the counts the model was fitted on, not a constant" do
      mostly_ordinary = model([:ordinary, :ordinary, :ordinary, :entity])
      mostly_entities = model([:ordinary, :entity, :entity, :entity])

      under = fn preferences ->
        score(EntityTypeScorer.expected_types("weather.query", preferences: preferences), preferences).p_ordinary
      end

      assert under.(mostly_ordinary) > under.(mostly_entities)
    end

    test "a reading the schema does not list is weighed by its data, not ruled out" do
      preferences = model()
      expected = EntityTypeScorer.expected_types("weather.query", preferences: preferences)

      refute "music_artist" in Expected.types(expected)
      likelihood = Expected.reading_likelihood(expected, "music_artist")

      assert likelihood > 0
      assert likelihood == SlotTypePreferences.reading_likelihood(expected.readings, "music_artist")
      assert p_entity(score(expected, preferences), "music_artist") > 0
    end
  end

  describe "failing loudly" do
    test "expected types that are not what expected_types/2 returns raise" do
      assert_raise ArgumentError, ~r/must be what expected_types\/2 or domain_expected_types\/2 returns/, fn ->
        score(%{"sys_date" => 1.0}, model())
      end
    end

    test "a candidate whose type the model cannot weigh raises under a context" do
      preferences = model()
      expected = EntityTypeScorer.expected_types("music.play", preferences: preferences)

      assert_raise ArgumentError, ~r/"emoji" \(gold label "emoji"\) is not in the en model's universe/, fn ->
        EntityTypeScorer.score(
          [%{entity_type: "emoji", value: "Thursday"}],
          %{match: "Thursday", sentence_initial: true, expected_types: expected},
          usage: @usage,
          preferences: preferences
        )
      end
    end
  end
end
