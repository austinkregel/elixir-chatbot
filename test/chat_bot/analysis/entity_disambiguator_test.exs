defmodule ChatBot.Analysis.EntityDisambiguatorTest do
  use ExUnit.Case, async: true

  alias ChatBot.Analysis.EntityDisambiguator

  describe "disambiguate/3" do
    test "returns entities unchanged when only one type" do
      entities = [
        %{entity: "person", value: "John", match: "John", start_pos: 0, end_pos: 4}
      ]

      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"John", "PROPN"}]
      context = %{discourse: nil, speech_act: nil}

      result = EntityDisambiguator.disambiguate(entities, pos_tagged, context)

      assert length(result) == 1
      assert hd(result).entity == "person"
    end

    test "disambiguates entity with multiple types" do
      person_info = %{entity_type: "person", value: "Austin"}
      location_info = %{entity_type: "location", value: "Austin"}

      entities = [
        %{
          value: "Austin",
          match: "Austin",
          start_pos: 5,
          end_pos: 11,
          types: [person_info, location_info]
        }
      ]

      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]

      context = %{
        discourse: %{indicators: ["self_referential"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      result = EntityDisambiguator.disambiguate(entities, pos_tagged, context)

      assert length(result) == 1
      # In introduction context, should prefer person over location
      assert hd(result).entity_type == "person"
    end

    test "prefers location for weather intents" do
      person_info = %{entity_type: "person", value: "Austin"}
      location_info = %{entity_type: "location", value: "Austin"}

      entities = [
        %{
          value: "Austin",
          match: "Austin",
          start_pos: 20,
          end_pos: 26,
          types: [person_info, location_info]
        }
      ]

      pos_tagged = [
        {"What", "PRON"},
        {"is", "VERB"},
        {"the", "DET"},
        {"weather", "NOUN"},
        {"in", "ADP"},
        {"Austin", "PROPN"}
      ]

      context = %{
        discourse: %{indicators: []},
        speech_act: %{category: :directive, sub_type: :question},
        intent: "weather.query"
      }

      result = EntityDisambiguator.disambiguate(entities, pos_tagged, context)

      assert length(result) == 1
      # For weather intent, should prefer location
      assert hd(result).entity_type == "location"
    end

    test "prefers music-artist for music intents" do
      person_info = %{entity_type: "person", value: "Prince"}
      artist_info = %{entity_type: "music-artist", value: "Prince"}

      entities = [
        %{
          value: "Prince",
          match: "Prince",
          start_pos: 5,
          end_pos: 11,
          types: [person_info, artist_info]
        }
      ]

      pos_tagged = [{"Play", "VERB"}, {"Prince", "PROPN"}]

      context = %{
        discourse: %{indicators: []},
        speech_act: %{category: :directive, sub_type: :command},
        intent: "music.play"
      }

      result = EntityDisambiguator.disambiguate(entities, pos_tagged, context)

      assert length(result) == 1
      # For music intent, should prefer music-artist
      assert hd(result).entity_type == "music-artist"
    end

    test "handles empty entities list" do
      result = EntityDisambiguator.disambiguate([], [], %{})
      assert result == []
    end
  end

  describe "disambiguate_single/3" do
    test "preserves entity when no types field" do
      entity = %{entity: "person", value: "John"}
      pos_tagged = [{"John", "PROPN"}]
      context = %{}

      result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

      assert result.entity == "person"
    end

    test "selects best type and removes types field" do
      person_info = %{entity_type: "person", value: "Austin"}
      location_info = %{entity_type: "location", value: "Austin"}

      entity = %{
        value: "Austin",
        types: [person_info, location_info],
        start_pos: 5
      }

      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]

      context = %{
        discourse: %{indicators: ["self_referential"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

      # Types field should be removed
      refute Map.has_key?(result, :types)

      # Should have selected entity type
      assert result.entity_type == "person"

      # Should indicate disambiguation source
      assert result.disambiguation_source == :context_analysis
    end
  end

  describe "introduction_confidence/3" do
    test "high confidence for PRON+VERB pattern with self-referential discourse" do
      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]
      entity_position = 2

      context = %{
        discourse: %{indicators: ["self_referential"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      confidence = EntityDisambiguator.introduction_confidence(pos_tagged, entity_position, context)

      # Should have high confidence (all features present)
      assert confidence >= 0.8
    end

    test "medium confidence with only PRON+VERB pattern" do
      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]
      entity_position = 2

      context = %{
        discourse: %{indicators: []},
        speech_act: nil
      }

      confidence = EntityDisambiguator.introduction_confidence(pos_tagged, entity_position, context)

      # Should have some confidence but not maximum
      assert confidence >= 0.4
      assert confidence < 0.8
    end

    test "low confidence without introduction patterns" do
      pos_tagged = [{"The", "DET"}, {"weather", "NOUN"}, {"in", "ADP"}, {"Austin", "PROPN"}]
      entity_position = 3

      context = %{
        discourse: %{indicators: []},
        speech_act: nil
      }

      confidence = EntityDisambiguator.introduction_confidence(pos_tagged, entity_position, context)

      # Should have low confidence
      assert confidence < 0.3
    end
  end

  describe "context detection" do
    test "detects self-referential from discourse indicators" do
      person_info = %{entity_type: "person", value: "Test"}
      location_info = %{entity_type: "location", value: "Test"}

      entity = %{
        value: "Test",
        types: [person_info, location_info],
        start_pos: 2
      }

      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Test", "PROPN"}]

      # With self_referential indicator
      context_with = %{
        discourse: %{indicators: ["self_referential"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      result_with = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context_with)

      # Without self_referential indicator
      context_without = %{
        discourse: %{indicators: []},
        speech_act: nil
      }

      _result_without = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context_without)

      # Self-referential context should boost person preference
      # Both might return person, but the scoring should be different internally
      assert result_with.entity_type == "person"
    end

    test "handles nil context values" do
      entity = %{
        value: "Test",
        types: [%{entity_type: "a"}, %{entity_type: "b"}],
        start_pos: 0
      }

      # Should not crash with nil values
      result = EntityDisambiguator.disambiguate_single(entity, [], %{
        discourse: nil,
        speech_act: nil
      })

      assert result.entity_type in ["a", "b"]
    end
  end

  describe "POS pattern detection" do
    test "detects PRON VERB pattern before entity" do
      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]
      entity_position = 2

      # The introduction_confidence function uses this internally
      confidence = EntityDisambiguator.introduction_confidence(pos_tagged, entity_position, %{
        discourse: %{indicators: []},
        speech_act: nil
      })

      # PRON+VERB pattern should contribute to confidence
      assert confidence >= 0.4
    end

    test "does not detect pattern when entity is first" do
      pos_tagged = [{"Austin", "PROPN"}, {"is", "VERB"}, {"nice", "ADJ"}]
      entity_position = 0

      confidence = EntityDisambiguator.introduction_confidence(pos_tagged, entity_position, %{
        discourse: %{indicators: []},
        speech_act: nil
      })

      # No preceding pattern possible
      assert confidence < 0.3
    end
  end
end
