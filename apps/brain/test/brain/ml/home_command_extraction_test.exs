defmodule Brain.ML.HomeCommandExtractionTest do
  @moduledoc """
  Room and light extraction from home-automation commands.

  The room cases are read from the room entity file, so every room the data
  declares is checked, not a hand-picked few. The lights reading rests on
  the lexicon: "lights" is used as "light" is (WordNet's suffix rules), so
  the band called Lights does not outweigh the everyday word.
  """
  use ExUnit.Case, async: false

  alias Brain.ML.EntityExtractor

  defp entities(text), do: EntityExtractor.extract_entities(text)

  defp find(entities, match), do: Enum.find(entities, &(String.downcase(&1.match) == match))

  defp room_values do
    data_path = Application.fetch_env!(:brain, :ml) |> Keyword.fetch!(:training_data_path)

    Path.join([data_path, "entities", "room_entries_en.json"])
    |> File.read!()
    |> Jason.decode!()
    |> Enum.map(&String.downcase(&1["value"]))
    |> Enum.uniq()
  end

  describe "the commands that prompted this" do
    test "'Can you turn off the office lights' finds the office room and plural lights" do
      found = entities("Can you turn off the office lights")

      assert %{entity_type: "room", number: :singular} = find(found, "office")
      assert %{entity_type: "lights", number: :plural} = find(found, "lights")
    end

    test "'Turn off all the lights' quantifies every light, and 'all' is no entity" do
      found = entities("Turn off all the lights")

      assert %{entity_type: "lights", number: :plural, quantifier: :total} = find(found, "lights")
      refute find(found, "all")
    end
  end

  describe "quantifiers" do
    test "quantify the head of the phrase, not its modifier" do
      found = entities("Turn off all the kitchen lights")

      assert %{entity_type: "room", quantifier: nil} = find(found, "kitchen")
      assert %{entity_type: "lights", quantifier: :total} = find(found, "lights")
    end

    test "every total quantifier works, across the function words between" do
      for text <- ["turn off every light", "turn off each light", "turn off all of the lights"] do
        [light] = Enum.filter(entities(text), &(&1.entity_type == "lights"))
        assert light.quantifier == :total, text
      end
    end

    test "no quantifier, none recorded" do
      assert %{quantifier: nil} = find(entities("turn off the lights"), "lights")
    end
  end

  describe "rooms from the data" do
    test "every declared room is read as a room in a lights command" do
      rooms = room_values()
      assert length(rooms) > 10

      failures =
        for room <- rooms,
            found = entities("turn off the #{room} lights"),
            entity = find(found, room),
            entity == nil or entity.entity_type != "room",
            do: {room, entity && entity.entity_type}

      assert failures == [], "rooms not read as rooms: #{inspect(failures)}"
    end
  end

  describe "function words" do
    test "a closed-class word is not read as the name it happens to match" do
      # "Of" is a town in Turkey, in the city list.
      refute find(entities("tell me more of it please"), "of")
    end
  end
end
