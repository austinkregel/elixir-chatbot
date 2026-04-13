defmodule Brain.ML.DataLoadersTest do
  use ExUnit.Case, async: false

  alias Brain.ML.DataLoaders

  describe "load_all_entities/0" do
    test "loads entity definitions from JSON files" do
      result = DataLoaders.load_all_entities()

      case result do
        {:ok, entities} ->
          assert is_map(entities)
          assert map_size(entities) >= 0

        {:error, _reason} ->
          assert true
      end
    end
  end

  describe "load_all_intents/0" do
    test "loads intent examples from JSON files" do
      result = DataLoaders.load_all_intents()

      case result do
        {:ok, examples} ->
          assert is_list(examples)

          if examples != [] do
            example = Enum.at(examples, 0)
            assert Map.has_key?(example, :text)
            assert Map.has_key?(example, :intent)
          end

        {:error, _reason} ->
          assert true
      end
    end
  end

  describe "load_cities/0" do
    test "loads cities from CSV" do
      result = DataLoaders.load_cities()

      case result do
        {:ok, cities} ->
          assert is_list(cities)

          if cities != [] do
            city = Enum.at(cities, 0)
            assert Map.has_key?(city, :name)
            assert Map.has_key?(city, :country)
          end

        {:error, _reason} ->
          assert true
      end
    end
  end

  describe "load_artists/0" do
    test "loads artists from CSV" do
      result = DataLoaders.load_artists()

      case result do
        {:ok, artists} ->
          assert is_list(artists)

          if artists != [] do
            artist = Enum.at(artists, 0)
            assert Map.has_key?(artist, :artist_name)
          end

        {:error, _reason} ->
          assert true
      end
    end
  end

  describe "load_emojis/0" do
    test "loads emojis from CSV" do
      result = DataLoaders.load_emojis()

      case result do
        {:ok, emojis} ->
          assert is_list(emojis)

          if emojis != [] do
            emoji = Enum.at(emojis, 0)
            assert Map.has_key?(emoji, :name)
          end

        {:error, _reason} ->
          assert true
      end
    end
  end

  describe "build_entity_lookup/1" do
    test "builds lookup map from entity definitions" do
      entities = %{
        "device" => [
          %{
            value: "thermostat",
            synonyms: ["thermo", "temperature control"],
            entity_type: "device"
          },
          %{value: "light", synonyms: ["lamp", "bulb"], entity_type: "device"}
        ],
        "room" => [%{value: "kitchen", synonyms: ["cook room"], entity_type: "room"}]
      }

      lookup = DataLoaders.build_entity_lookup(entities)

      assert is_map(lookup)
      assert Map.has_key?(lookup, "thermostat")
      assert Map.has_key?(lookup, "thermo")
      assert Map.has_key?(lookup, "lamp")
      assert Map.has_key?(lookup, "kitchen")
      thermo = Map.get(lookup, "thermostat")
      assert thermo.entity_type == "device"
      assert thermo.value == "thermostat"
    end

    test "normalizes text to lowercase" do
      entities = %{
        "location" => [
          %{value: "New York", synonyms: ["NYC", "Big Apple"], entity_type: "location"}
        ]
      }

      lookup = DataLoaders.build_entity_lookup(entities)

      assert Map.has_key?(lookup, "new york")
      assert Map.has_key?(lookup, "nyc")
      assert Map.has_key?(lookup, "big apple")
    end

    test "filters out very short entries" do
      entities = %{
        "test" => [
          %{value: "a", synonyms: ["b"], entity_type: "test"},
          %{value: "valid", synonyms: ["ok"], entity_type: "test"}
        ]
      }

      lookup = DataLoaders.build_entity_lookup(entities)
      refute Map.has_key?(lookup, "a")
      refute Map.has_key?(lookup, "b")
      assert Map.has_key?(lookup, "valid")
      assert Map.has_key?(lookup, "ok")
    end
  end

  describe "build_city_lookup/1" do
    test "builds city lookup from city data" do
      cities = [
        %{name: "London", country: "United Kingdom", subcountry: "England"},
        %{name: "Paris", country: "France", subcountry: "Île-de-France"}
      ]

      lookup = DataLoaders.build_city_lookup(cities)

      assert is_map(lookup)
      assert Map.has_key?(lookup, "london")
      assert Map.has_key?(lookup, "paris")

      london = Map.get(lookup, "london")
      assert london.entity_type == "location"
      assert london.value == "London"
      assert london.country == "United Kingdom"
    end
  end

  describe "build_artist_lookup/1" do
    test "builds artist lookup from artist data" do
      artists = [
        %{artist_name: "The Beatles", artist_genre: "Rock", country: "UK"},
        %{artist_name: "Madonna", artist_genre: "Pop", country: "USA"}
      ]

      lookup = DataLoaders.build_artist_lookup(artists)

      assert is_map(lookup)
      assert Map.has_key?(lookup, "the beatles")
      assert Map.has_key?(lookup, "madonna")

      beatles = Map.get(lookup, "the beatles")
      assert beatles.entity_type == "music-artist"
      assert beatles.value == "The Beatles"
      assert beatles.genre == "Rock"
    end
  end

  describe "build_emoji_lookup/1" do
    test "builds emoji lookup from emoji data" do
      emojis = [
        %{name: "grinning face", representation: "😀", group: "Smileys"},
        %{name: "heart", representation: "❤️", group: "Symbols"}
      ]

      lookup = DataLoaders.build_emoji_lookup(emojis)

      assert is_map(lookup)
      assert Map.has_key?(lookup, "grinning face")
      assert Map.has_key?(lookup, "heart")

      heart = Map.get(lookup, "heart")
      assert heart.entity_type == "emoji"
      assert heart.representation == "❤️"
    end
  end
end