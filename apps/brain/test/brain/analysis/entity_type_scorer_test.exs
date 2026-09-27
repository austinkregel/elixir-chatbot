defmodule Brain.Analysis.EntityTypeScorerTest do
  @moduledoc """
  What a gazetteer match means is decided by evidence, not by rules naming
  words or types: how the word is used (usage facts in the brain's lexicon,
  seeded from SemCor), how it was typed, and which entity types the intent's
  slot schema lists.

  Beyond the two sentences the scorer was built for, every test here takes
  its words, types and intents from data -- the entity files, the intent
  registry, the seeded lexicon -- or changes the evidence and checks the
  answer moves with it. A rule written for "thursday" could not pass them.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Analysis.{EntityTypeScorer, SlotDetector}
  alias Brain.Lexicon.UserDefined
  alias Brain.ML.{EntityExtractor, Gazetteer}

  @config Application.compile_env!(:brain, :entity_type_scoring)

  defp entity_for(text, match, opts \\ []) do
    text
    |> EntityExtractor.extract_entities(opts)
    |> Enum.find(&(String.downcase(&1.match) == match))
  end

  defp candidates(word) do
    case Gazetteer.lookup(word) do
      {:ok, infos} when is_list(infos) -> infos
      {:ok, info} when is_map(info) -> [info]
    end
  end

  defp score(word, typed, opts \\ []) do
    {evidence_opts, score_opts} = Keyword.split(opts, [:sentence_initial, :expected_types])

    evidence = %{
      match: typed,
      sentence_initial: Keyword.get(evidence_opts, :sentence_initial, false),
      expected_types: Keyword.get(evidence_opts, :expected_types)
    }

    EntityTypeScorer.score(candidates(word), evidence, score_opts)
  end

  describe "the sentences it was built for" do
    test "with no context, 'next thursday I need to call my dad' reads thursday as a date" do
      entity = entity_for("next thursday I need to call my dad", "thursday")

      assert entity, "thursday was not extracted"
      assert entity.entity_type == "sys_date"
    end

    test "the call intent leaves thursday a date" do
      # communication.call lists only person. Neither the date nor the band is
      # listed, so the intent is no evidence between them and usage decides.
      entity =
        entity_for("next thursday I need to call my dad", "thursday", intent: "communication.call")

      assert entity.entity_type == "sys_date"
    end

    test "'play thursday on spotify' under music.play reads thursday as vaguely more like an artist" do
      entity = entity_for("play thursday on spotify", "thursday", intent: "music.play")

      assert entity.entity_type == "music_artist"
      assert entity.type_posteriors["music_artist"] > entity.type_posteriors["sys_date"]

      # Vaguely: the date stays a live reading, because thursday is used as a
      # date and never as a band.
      assert entity.type_posteriors["sys_date"] > 0.1
    end

    test "capitalizing Thursday makes the artist reading firmer" do
      lower = entity_for("play thursday on spotify", "thursday", intent: "music.play")
      upper = entity_for("play Thursday on spotify", "thursday", intent: "music.play")

      assert upper.type_posteriors["music_artist"] > lower.type_posteriors["music_artist"]
    end

    test "the pass-2 rescore moves an extracted entity once the intent is known, and back" do
      extracted = entity_for("play thursday on spotify", "thursday")
      assert extracted.entity_type == "sys_date"

      [as_music] = EntityTypeScorer.rescore([extracted], "music.play")
      assert as_music.entity_type == "music_artist"

      # The candidates survive rescoring, so it is not a one-way door.
      [again] = EntityTypeScorer.rescore([as_music], nil)
      assert again.entity_type == "sys_date"
    end
  end

  describe "readings follow the data, not a word list" do
    test "every weekday and month the date file lists reads as a date with no context" do
      # Every sys_date entry the lexicon records as used as a date, read from
      # the entity file and the seeded lexicon.
      data_path = Application.fetch_env!(:brain, :ml) |> Keyword.fetch!(:training_data_path)

      words =
        Path.join([data_path, "entities", "sys_date_entries_en.json"])
        |> File.read!()
        |> Jason.decode!()
        |> Enum.flat_map(&[&1["value"] | &1["synonyms"]])
        |> Enum.map(&String.downcase/1)
        |> Enum.uniq()
        |> Enum.filter(&(EntityTypeScorer.usage(&1).types["sys_date"] != nil))

      assert length(words) >= 12, "expected the weekdays and months, got #{inspect(words)}"

      for word <- words do
        entity = entity_for("remind me on #{word} please", word)
        assert entity, "#{word} was not extracted"
        assert entity.entity_type == "sys_date", "#{word} read as #{entity.entity_type}"
      end
    end

    test "any intent that lists music_artist and not sys_date reads thursday as the artist" do
      intents = intents_listing("music_artist", "sys_date")
      assert intents != []

      for intent <- intents do
        assert score("thursday", "thursday", expected_types: EntityTypeScorer.expected_types(intent)).entity_type ==
                 "music_artist",
               "under #{intent}"
      end
    end

    test "any intent that lists sys_date and not music_artist reads thursday as the date" do
      intents = intents_listing("sys_date", "music_artist")
      assert intents != []

      for intent <- intents do
        assert score("thursday", "thursday", expected_types: EntityTypeScorer.expected_types(intent)).entity_type ==
                 "sys_date",
               "under #{intent}"
      end
    end

    test "running statistics move the answer: enough observed band uses make thursday a band" do
      prefix = :"scorer_test_#{System.unique_integer([:positive])}"
      store = :"#{prefix}_store"
      start_supervised!({UserDefined, name: store, table_prefix: prefix}, id: store)

      assert score("thursday", "thursday", lexicon: store).entity_type == "sys_date"

      {:ok, _} =
        UserDefined.put_fact(
          %{
            word: "thursday",
            kind: "property",
            key: "sense_usage",
            ref: "music_artist",
            value: %{"count" => 100},
            source: "derived"
          },
          store
        )

      assert score("thursday", "thursday", lexicon: store).entity_type == "music_artist"
    end

    test "the prior is exactly the usage it is given" do
      # Same candidates, same typing, no context: only the counts differ.
      as_date = score("thursday", "thursday", usage: %{types: %{"sys_date" => 5}, ordinary: %{}})
      as_band = score("thursday", "thursday", usage: %{types: %{"music_artist" => 5}, ordinary: %{}})

      assert as_date.entity_type == "sys_date"
      assert as_band.entity_type == "music_artist"
    end
  end

  describe "an ordinary word is a reading too" do
    test "'have a nice day' does not find a place" do
      refute entity_for("have a nice day", "nice")
    end

    test "'the weather in Nice' does, from the capital letter alone" do
      entity = entity_for("what's the weather in Nice", "nice")

      assert entity, "Nice was not extracted"
      assert entity.entity_type == "location"
    end

    test "a capital that starts the sentence says nothing" do
      first = score("nice", "Nice", sentence_initial: true)
      lower = score("nice", "nice", sentence_initial: false)
      mid = score("nice", "Nice", sentence_initial: false)

      # Mid-sentence, the capital is strong evidence for the city; starting a
      # sentence, it is none, and "Nice" reads as the adjective it usually is.
      assert first.p_ordinary > mid.p_ordinary
      assert first.p_ordinary > 0.5
      # A lowercase "nice" mid-sentence is evidence against the name, so it
      # leans further toward the adjective than an uninformative capital does.
      assert lower.p_ordinary > first.p_ordinary
    end

    test "a name WordNet has never seen is not doubted for it" do
      # Every artist candidate WordNet has no counted use of keeps the whole of
      # its base confidence.
      entity = entity_for("play coldplay", "coldplay")

      assert entity.entity_type == "music_artist"
      assert entity.p_ordinary == 0.0
      assert entity.confidence == EntityTypeScorer.base_confidence("coldplay", "music_artist")
    end

    test "a word only a name list offers, and used commonly, is dropped" do
      # "yes" is a band, and an everyday word used far more often.
      refute entity_for("yes please", "yes")
    end

    test "a curated common word keeps its everyday meaning" do
      # A date file listing "later" means the adverb.
      entity = entity_for("I may go out later", "later")

      assert entity.entity_type == "sys_date"
      assert entity.p_ordinary == 0.0
    end
  end

  describe "part of speech" do
    # Tags are given explicitly: these test how the scorer weighs a tag, not
    # how accurate the tagger is.
    defp tagged(word, typed, tag, opts \\ []) do
      evidence = %{match: typed, sentence_initial: false, expected_types: nil, tag: tag}
      EntityTypeScorer.score(candidates(word), evidence, opts)
    end

    test "tagged as a verb, 'may' is the modal, not the month" do
      assert tagged("may", "may", "VERB").p_ordinary > 0.9
    end

    test "the month reading of 'may' is likelier tagged NOUN than tagged VERB" do
      assert tagged("may", "may", "NOUN").posteriors["sys_date"] >
               tagged("may", "may", "VERB").posteriors["sys_date"]
    end

    test "tagged as an adjective, 'light' is an ordinary word; as a noun, the light" do
      as_adj = tagged("light", "light", "ADJ")
      as_noun = tagged("light", "light", "NOUN")

      assert as_adj.p_ordinary > 0.5
      assert as_noun.entity_type == "lights"
      assert as_noun.p_ordinary < 0.2
    end

    test "a curated word takes only the uses in the parts of speech its mentions take" do
      # With no tag, 'light' the device competes with the adjective and verb
      # uses, which the lights file does not mean.
      untagged = tagged("light", "light", nil)
      assert untagged.p_ordinary > 0.2
    end

    test "no tag says nothing: the same as leaving the tag out" do
      with_nil = tagged("thursday", "thursday", nil)

      without =
        EntityTypeScorer.score(candidates("thursday"), %{match: "thursday", sentence_initial: false})

      assert with_nil == without
    end

    test "with tag_mismatch_likelihood at 1, the tag stops mattering" do
      blind = Keyword.put(@config, :tag_mismatch_likelihood, 1.0)

      assert tagged("may", "may", "VERB", config: blind) == tagged("may", "may", "NOUN", config: blind)
    end

    test "a stale or unknown ordinary-use ref fails the score rather than being summed" do
      assert_raise RuntimeError, ~r/unknown ordinary_usage ref "other_pos"/, fn ->
        EntityTypeScorer.score(candidates("nice"), %{match: "nice", sentence_initial: false},
          usage: %{types: %{}, ordinary: %{"other_pos" => 29}}
        )
      end
    end
  end

  describe "inflected forms are weighed by their lemma's uses" do
    test "a plural has the usage of its lemma, which it has none of itself" do
      assert Brain.Lexicon.UserDefined.facts("lights", kind: "property") == []

      lights = EntityTypeScorer.usage("lights")
      light = EntityTypeScorer.usage("light")

      assert lights == light
      assert Enum.sum(Map.values(lights.ordinary)) > 0
    end

    test "a band named for a plural noun loses to the noun, with no curated entry to help" do
      # Only the artist candidate: the situation for any plural name the
      # curated files do not list.
      band = Enum.filter(candidates("lights"), &(&1[:source] == "artist"))
      assert [%{entity_type: "music_artist"}] = band

      result =
        EntityTypeScorer.score(band, %{match: "lights", sentence_initial: false, expected_types: nil})

      assert result.p_ordinary > 0.9
    end
  end

  describe "confidence" do
    test "readings that tie do not halve each other" do
      entity = entity_for("open the door", "door")
      posteriors = Map.values(entity.type_posteriors)

      assert length(posteriors) > 1
      assert Enum.max(posteriors) < 0.5
      assert entity.confidence == EntityTypeScorer.base_confidence("door", entity.entity_type)
    end
  end

  describe "the constants shift the answer in the direction they claim" do
    test "with context_mismatch_likelihood at 1, the intent stops mattering" do
      music = EntityTypeScorer.expected_types("music.play")
      indifferent = Keyword.put(@config, :context_mismatch_likelihood, 1.0)

      assert score("thursday", "thursday", expected_types: music).entity_type == "music_artist"
      assert score("thursday", "thursday", expected_types: music, config: indifferent).entity_type == "sys_date"
    end

    test "a smaller context_mismatch_likelihood makes the intent weigh more" do
      music = EntityTypeScorer.expected_types("music.play")
      strict = Keyword.put(@config, :context_mismatch_likelihood, @config[:context_mismatch_likelihood] / 10)

      assert score("thursday", "thursday", expected_types: music, config: strict).posteriors["music_artist"] >
               score("thursday", "thursday", expected_types: music).posteriors["music_artist"]
    end

    test "when names and ordinary words are capitalized alike, casing says nothing" do
      blind = Keyword.put(@config, :ordinary_word_capitalized, @config[:proper_name_capitalized])

      mid = score("nice", "Nice", config: blind)
      first = score("nice", "Nice", sentence_initial: true, config: blind)

      assert_in_delta mid.p_ordinary, first.p_ordinary, 1.0e-12
    end

    test "a larger prior_smoothing washes the counts toward even" do
      heavy = Keyword.put(@config, :prior_smoothing, 10_000.0)

      assert score("thursday", "thursday", config: heavy).posteriors["sys_date"] <
               score("thursday", "thursday").posteriors["sys_date"]
    end
  end

  describe "failing loudly" do
    test "no candidates is an error, not an empty answer" do
      assert_raise ArgumentError, ~r/no candidates/, fn ->
        EntityTypeScorer.score([], %{match: "zorbl", sentence_initial: false})
      end
    end

    test "a candidate without a type is an error" do
      assert_raise ArgumentError, ~r/no entity type/, fn ->
        EntityTypeScorer.score([%{value: "zorbl"}], %{match: "zorbl", sentence_initial: false}, usage: %{})
      end
    end

    test "an entity without its candidates cannot be scored" do
      assert_raise ArgumentError, ~r/cannot score/, fn ->
        EntityTypeScorer.apply_to(%{entity_type: "sys_date", match: "thursday"}, nil)
      end
    end
  end

  describe "sentence_initial?/2" do
    test "the first word of the text or of any sentence in it" do
      assert EntityTypeScorer.sentence_initial?("Nice is lovely", 0)
      assert EntityTypeScorer.sentence_initial?("Hi.  Nice is lovely", 5)
      assert EntityTypeScorer.sentence_initial?("Really? Nice!", 8)
      refute EntityTypeScorer.sentence_initial?("I love Nice", 7)
      refute EntityTypeScorer.sentence_initial?("weather, Nice", 9)
    end
  end

  # Intents whose slot schema lists `wanted` and not `unwanted`.
  defp intents_listing(wanted, unwanted) do
    SlotDetector.list_schemas()
    |> Map.keys()
    |> Enum.filter(fn intent ->
      types = EntityTypeScorer.expected_types(intent)
      wanted in types and unwanted not in types
    end)
    |> Enum.sort()
  end
end
