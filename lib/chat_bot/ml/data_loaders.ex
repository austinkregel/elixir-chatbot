defmodule ChatBot.ML.DataLoaders do
  @moduledoc """
  Data loading utilities for training data from various sources.

  Supports:
  - CSV files (cities, artists, emojis)
  - JSON files (entities, intents in Dialogflow format)
  - Entity normalization and standardization
  """

  require Logger

  @type entity_entry :: %{
          value: String.t(),
          synonyms: [String.t()],
          entity_type: String.t()
        }

  @type intent_example :: %{
          text: String.t(),
          intent: String.t(),
          entities: [%{text: String.t(), type: String.t(), alias: String.t()}]
        }

  # ============================================================================
  # CSV Loading
  # ============================================================================

  @doc """
  Load world cities from CSV file.
  Returns a list of city entries with name, country, subcountry, and geonameid.
  """
  def load_cities(path \\ nil) do
    path = path || get_data_path("world-cities.csv")

    case File.read(path) do
      {:ok, content} ->
        cities = parse_csv(content, [:name, :country, :subcountry, :geonameid])
        Logger.info("Loaded cities", %{count: length(cities)})
        {:ok, cities}

      {:error, reason} ->
        Logger.warning("Failed to load cities", %{path: path, reason: reason})
        {:error, reason}
    end
  end

  @doc """
  Load US cities from CSV file (comprehensive dataset with ~30k cities).
  Returns a list of city entries with city, state_code, state_name, county, latitude, longitude.
  """
  def load_us_cities(path \\ nil) do
    path = path || get_data_path("us_cities.csv")

    case File.read(path) do
      {:ok, content} ->
        cities =
          parse_csv(content, [
            :id,
            :state_code,
            :state_name,
            :city,
            :county,
            :latitude,
            :longitude
          ])

        Logger.info("Loaded US cities", %{count: length(cities)})
        {:ok, cities}

      {:error, reason} ->
        Logger.warning("Failed to load US cities", %{path: path, reason: reason})
        {:error, reason}
    end
  end

  @doc """
  Load music artists from CSV file.
  Returns a list of artist entries with name, genre, country, etc.
  """
  def load_artists(path \\ nil) do
    path = path || get_data_path("Global Music Artists.csv")

    case File.read(path) do
      {:ok, content} ->
        artists =
          parse_csv(content, [:artist_name, :artist_genre, :artist_img, :artist_id, :country])

        Logger.info("Loaded artists", %{count: length(artists)})
        {:ok, artists}

      {:error, reason} ->
        Logger.warning("Failed to load artists", %{path: path, reason: reason})
        {:error, reason}
    end
  end

  @doc """
  Load emoji definitions from CSV file.
  Returns a list of emoji entries with group, subgroup, representation, name, etc.
  """
  def load_emojis(path \\ nil) do
    path = path || get_data_path("emojis.csv")

    case File.read(path) do
      {:ok, content} ->
        emojis =
          parse_csv(content, [
            :group,
            :subgroup,
            :codepoint,
            :status,
            :representation,
            :name,
            :section
          ])

        Logger.info("Loaded emojis", %{count: length(emojis)})
        {:ok, emojis}

      {:error, reason} ->
        Logger.warning("Failed to load emojis", %{path: path, reason: reason})
        {:error, reason}
    end
  end

  # ============================================================================
  # JSON Entity Loading
  # ============================================================================

  @doc """
  Load all entity definitions from the entities directory.
  Returns a map of entity_type => list of entity entries.
  """
  def load_all_entities(path \\ nil) do
    entities_dir = path || get_data_path("entities")

    case File.ls(entities_dir) do
      {:ok, files} ->
        entities =
          files
          |> Enum.filter(&String.ends_with?(&1, ".json"))
          |> Enum.reduce(%{}, fn file, acc ->
            entity_type = extract_entity_type(file)
            file_path = Path.join(entities_dir, file)

            case load_entity_file(file_path, entity_type) do
              {:ok, entries} ->
                existing = Map.get(acc, entity_type, [])
                Map.put(acc, entity_type, existing ++ entries)

              {:error, _} ->
                acc
            end
          end)

        total_entries = entities |> Map.values() |> Enum.map(&length/1) |> Enum.sum()

        Logger.info("Loaded entity definitions", %{
          entity_types: map_size(entities),
          total_entries: total_entries
        })

        {:ok, entities}

      {:error, reason} ->
        Logger.warning("Failed to list entities directory", %{path: entities_dir, reason: reason})
        {:error, reason}
    end
  end

  @doc """
  Load a single entity definition file.
  Supports both _entries_en.json format and regular .json format.
  """
  def load_entity_file(path, entity_type) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            entries = parse_entity_data(data, entity_type)
            {:ok, entries}

          {:error, reason} ->
            Logger.debug("Failed to parse entity file", %{path: path, reason: reason})
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ============================================================================
  # Intent Loading with Entity Annotations
  # ============================================================================

  @doc """
  Load all intent training data from the intents directory.
  Returns a list of intent examples with text, intent label, and entity annotations.
  """
  def load_all_intents(path \\ nil) do
    intents_dir = path || get_data_path("intents")

    case File.ls(intents_dir) do
      {:ok, files} ->
        json_files = Enum.filter(files, &String.ends_with?(&1, ".json"))

        examples =
          Enum.flat_map(json_files, fn file ->
            file_path = Path.join(intents_dir, file)
            intent_name = extract_intent_name(file)

            case load_intent_file(file_path, intent_name) do
              {:ok, file_examples} -> file_examples
              {:error, _} -> []
            end
          end)

        Logger.info("Loaded intent examples", %{
          files: length(json_files),
          examples: length(examples)
        })

        {:ok, examples}

      {:error, reason} ->
        Logger.warning("Failed to list intents directory", %{path: intents_dir, reason: reason})
        {:error, reason}
    end
  end

  @doc """
  Load a single intent file and extract training examples with entity annotations.
  """
  def load_intent_file(path, intent_name) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            examples = parse_intent_data(data, intent_name)
            {:ok, examples}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ============================================================================
  # Smalltalk Responses Loading
  # ============================================================================

  @doc """
  Load custom smalltalk responses.
  Returns a map of action => list of response strings.
  """
  def load_smalltalk_responses(path \\ nil) do
    path = path || get_data_path("customSmalltalkResponses_en.json")

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            responses =
              Enum.reduce(data, %{}, fn item, acc ->
                action = Map.get(item, "action")
                answers = Map.get(item, "customAnswers", [])

                if is_binary(action) and is_list(answers) do
                  Map.put(acc, action, answers)
                else
                  acc
                end
              end)

            Logger.info("Loaded smalltalk responses", %{actions: map_size(responses)})
            {:ok, responses}

          {:ok, data} when is_map(data) ->
            {:ok, data}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        Logger.warning("Failed to load smalltalk responses", %{path: path, reason: reason})
        {:error, reason}
    end
  end

  # ============================================================================
  # Entity Normalization
  # ============================================================================

  @doc """
  Build a normalized lookup map from entity entries.
  Maps lowercase synonym -> {entity_type, canonical_value}
  """
  def build_entity_lookup(entities) when is_map(entities) do
    Enum.reduce(entities, %{}, fn {entity_type, entries}, acc ->
      Enum.reduce(entries, acc, fn entry, inner_acc ->
        value = entry.value
        synonyms = entry.synonyms

        # Add all synonyms (including the value itself) to the lookup
        Enum.reduce([value | synonyms], inner_acc, fn synonym, lookup ->
          normalized = normalize_text(synonym)

          if String.length(normalized) >= 2 do
            Map.put(lookup, normalized, %{
              entity_type: entity_type,
              value: value,
              original: synonym
            })
          else
            lookup
          end
        end)
      end)
    end)
  end

  @doc """
  Build city lookup from loaded city data.
  Maps lowercase city name -> city info
  """
  def build_city_lookup(cities) when is_list(cities) do
    Enum.reduce(cities, %{}, fn city, acc ->
      name = Map.get(city, :name) || ""
      normalized = normalize_text(name)

      if String.length(normalized) >= 2 do
        Map.put(acc, normalized, %{
          entity_type: "location",
          value: name,
          country: Map.get(city, :country),
          subcountry: Map.get(city, :subcountry)
        })
      else
        acc
      end
    end)
  end

  # Common English words that happen to be city names - skip standalone matching
  @ambiguous_city_names ~w(
    tell me you can the and for in on at to be is are was were
    will would could should have has had do does did may might
    can could shall should will would be being been
    home big little new old good bad high low long short
    sun moon star lake river hill dale view park spring
    point bay city town fair hope love joy grace faith
    burns wells ford bridge mills dale glen grove
  )

  @doc """
  Build US city lookup from loaded US city data.
  Maps lowercase city name -> city info with state.
  Also creates entries for "city, state" format.
  Filters out ambiguous city names that are common English words.
  """
  def build_us_city_lookup(cities) when is_list(cities) do
    Enum.reduce(cities, %{}, fn city, acc ->
      name = Map.get(city, :city) || ""
      state_code = Map.get(city, :state_code) || ""
      state_name = Map.get(city, :state_name) || ""
      county = Map.get(city, :county) || ""
      normalized = normalize_text(name)

      # Skip very short names or common English words for standalone matching
      is_ambiguous = String.length(normalized) <= 3 or normalized in @ambiguous_city_names

      if String.length(normalized) >= 2 do
        city_info = %{
          entity_type: "location",
          type: "city",
          value: name,
          country: "United States",
          state_code: state_code,
          state_name: state_name,
          county: county,
          region: state_name
        }

        # For ambiguous names, only add with state qualifier (not standalone)
        acc =
          if is_ambiguous do
            acc
          else
            Map.put(acc, normalized, city_info)
          end

        acc
        # City, State Code with comma (e.g., "owosso, mi")
        |> Map.put("#{normalized}, #{String.downcase(state_code)}", city_info)
        # City, State Name with comma (e.g., "owosso, michigan")
        |> Map.put("#{normalized}, #{String.downcase(state_name)}", city_info)
        # City State Code without comma (e.g., "owosso mi")
        |> Map.put("#{normalized} #{String.downcase(state_code)}", city_info)
        # City State Name without comma (e.g., "owosso michigan")
        |> Map.put("#{normalized} #{String.downcase(state_name)}", city_info)
      else
        acc
      end
    end)
  end

  @doc """
  Build artist lookup from loaded artist data.
  Maps lowercase artist name -> artist info
  """
  def build_artist_lookup(artists) when is_list(artists) do
    Enum.reduce(artists, %{}, fn artist, acc ->
      name = Map.get(artist, :artist_name) || ""
      normalized = normalize_text(name)

      if String.length(normalized) >= 2 do
        Map.put(acc, normalized, %{
          entity_type: "music-artist",
          value: name,
          genre: Map.get(artist, :artist_genre),
          country: Map.get(artist, :country)
        })
      else
        acc
      end
    end)
  end

  @doc """
  Build emoji lookup from loaded emoji data.
  Maps lowercase emoji name -> emoji info
  """
  def build_emoji_lookup(emojis) when is_list(emojis) do
    Enum.reduce(emojis, %{}, fn emoji, acc ->
      name = Map.get(emoji, :name) || ""
      normalized = normalize_text(name)

      if String.length(normalized) >= 2 do
        Map.put(acc, normalized, %{
          entity_type: "emoji",
          value: name,
          representation: Map.get(emoji, :representation),
          group: Map.get(emoji, :group)
        })
      else
        acc
      end
    end)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_data_path(filename) do
    base_path = Application.get_env(:chat_bot, :ml)[:training_data_path] || "data"
    Path.join(base_path, filename)
  end

  defp parse_csv(content, headers) do
    lines = String.split(content, "\n", trim: true)

    case lines do
      [_header_line | data_lines] ->
        Enum.map(data_lines, fn line ->
          values = parse_csv_line(line)

          headers
          |> Enum.zip(values)
          |> Enum.into(%{})
        end)
        |> Enum.filter(fn row ->
          # Filter out empty rows
          Enum.any?(row, fn {_k, v} -> v != "" and v != nil end)
        end)

      _ ->
        []
    end
  end

  defp parse_csv_line(line) do
    # Simple CSV parsing that handles quoted fields
    parse_csv_fields(line, [], "", false)
  end

  defp parse_csv_fields("", acc, current, _in_quotes) do
    Enum.reverse([String.trim(current) | acc])
  end

  defp parse_csv_fields(<<"\"", rest::binary>>, acc, current, false) do
    # Start of quoted field
    parse_csv_fields(rest, acc, current, true)
  end

  defp parse_csv_fields(<<"\"\"", rest::binary>>, acc, current, true) do
    # Escaped quote inside quoted field
    parse_csv_fields(rest, acc, current <> "\"", true)
  end

  defp parse_csv_fields(<<"\"", rest::binary>>, acc, current, true) do
    # End of quoted field
    parse_csv_fields(rest, acc, current, false)
  end

  defp parse_csv_fields(<<",", rest::binary>>, acc, current, false) do
    # Field separator (not in quotes)
    parse_csv_fields(rest, [String.trim(current) | acc], "", false)
  end

  defp parse_csv_fields(<<char::utf8, rest::binary>>, acc, current, in_quotes) do
    parse_csv_fields(rest, acc, current <> <<char::utf8>>, in_quotes)
  end

  defp extract_entity_type(filename) do
    filename
    |> String.replace("_entries_en.json", "")
    |> String.replace(".json", "")
    |> String.replace("-", "_")
  end

  defp parse_entity_data(data, entity_type) when is_list(data) do
    Enum.map(data, fn item ->
      value = Map.get(item, "value") || Map.get(item, "name") || ""
      synonyms = Map.get(item, "synonyms", [])

      %{
        value: value,
        synonyms: List.wrap(synonyms),
        entity_type: entity_type
      }
    end)
    |> Enum.filter(fn entry -> entry.value != "" end)
  end

  defp parse_entity_data(%{"entries" => entries}, entity_type) when is_list(entries) do
    parse_entity_data(entries, entity_type)
  end

  defp parse_entity_data(_, _entity_type), do: []

  defp extract_intent_name(filename) do
    filename
    |> String.replace("_usersays_en.json", "")
    |> String.replace("_usersays.json", "")
    |> String.replace(".json", "")
    |> String.replace(" - ", ".")
    |> String.replace(" ", ".")
    |> normalize_intent_name()
  end

  defp normalize_intent_name(name) do
    # Remove context annotations like "context_heating"
    name
    |> String.replace(~r/\s*-\s*context_\w+/, "")
    |> String.replace(~r/\s*-\s*comment_.*$/, "")
    |> String.trim()
  end

  defp parse_intent_data(data, intent_name) when is_list(data) do
    Enum.flat_map(data, fn example ->
      case extract_example_with_entities(example) do
        {:ok, text, entities} when text != "" ->
          [
            %{
              text: text,
              intent: intent_name,
              entities: entities
            }
          ]

        _ ->
          []
      end
    end)
  end

  defp parse_intent_data(%{"userSays" => examples}, intent_name) when is_list(examples) do
    parse_intent_data(examples, intent_name)
  end

  defp parse_intent_data(%{"responses" => responses}, intent_name) when is_list(responses) do
    # Extract from response messages (for non-usersays files)
    Enum.flat_map(responses, fn resp ->
      msgs = Map.get(resp, "messages", [])

      Enum.flat_map(msgs, fn msg ->
        speech = Map.get(msg, "speech")

        cond do
          is_binary(speech) ->
            [%{text: speech, intent: intent_name, entities: []}]

          is_list(speech) ->
            Enum.map(speech, &%{text: &1, intent: intent_name, entities: []})

          true ->
            []
        end
      end)
    end)
  end

  defp parse_intent_data(_, _intent_name), do: []

  defp extract_example_with_entities(example) do
    case Map.get(example, "data") do
      nil ->
        # Simple text field
        text = Map.get(example, "text", "")
        {:ok, text, []}

      data when is_list(data) ->
        # Dialogflow format with entity annotations
        {text, entities, _pos} =
          Enum.reduce(data, {"", [], 0}, fn item, {acc_text, acc_entities, pos} ->
            item_text = Map.get(item, "text", "")
            meta = Map.get(item, "meta")
            alias_name = Map.get(item, "alias")

            new_pos = pos + String.length(item_text)

            if meta != nil and alias_name != nil do
              entity = %{
                text: item_text,
                type: normalize_meta_type(meta),
                alias: alias_name,
                start_pos: pos,
                end_pos: new_pos - 1
              }

              {acc_text <> item_text, [entity | acc_entities], new_pos}
            else
              {acc_text <> item_text, acc_entities, new_pos}
            end
          end)

        {:ok, String.trim(text), Enum.reverse(entities)}

      _ ->
        {:error, :invalid_format}
    end
  end

  defp normalize_meta_type(meta) do
    meta
    |> String.replace("@sys.", "")
    |> String.replace("@", "")
    |> String.replace("-", "_")
  end

  defp normalize_text(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_text(_), do: ""
end
