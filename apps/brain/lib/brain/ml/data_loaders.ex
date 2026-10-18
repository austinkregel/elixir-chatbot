defmodule Brain.ML.DataLoaders do
  @moduledoc "Data loading utilities for training data from various sources.\n\nSupports:\n- CSV files (cities, artists, emojis)\n- JSON files (entities, intents in Dialogflow format)\n- Entity normalization and standardization\n"

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

  @doc "Load world cities from CSV file.\nReturns a list of city entries with name, country, subcountry, and geonameid.\n"
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

  @doc "Load US cities from CSV file (comprehensive dataset with ~30k cities).\nReturns a list of city entries with city, state_code, state_name, county, latitude, longitude.\n"
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

  @doc "Load music artists from CSV file.\nReturns a list of artist entries with name, genre, country, etc.\n"
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

  @doc "Load emoji definitions from CSV file.\nReturns a list of emoji entries with group, subgroup, representation, name, etc.\n"
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

  @doc "Load all entity definitions from the entities directory.\nReturns a map of entity_type => list of entity entries.\n"
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

              {:error, reason} ->
                raise "DataLoaders: entity file #{file_path} could not be loaded: #{inspect(reason)}"
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
  Unwraps a loader result for a source the caller cannot work without.

  Raises when the source failed to load, naming the source and the data
  directory, so a gazetteer is never built from fewer sources than it expects.
  """
  def require_source!({:ok, records}, _source), do: records

  def require_source!({:error, reason}, source) do
    raise "DataLoaders: required source #{inspect(source)} could not be loaded from " <>
            "#{get_data_path("")}: #{inspect(reason)}"
  end

  @doc "Load a single entity definition file.\nSupports both _entries_en.json format and regular .json format.\n"
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

  @doc "Load all intent training data.\n\nFirst tries the consolidated gold standard (preferred), then falls back to\nlegacy data/intents directory if gold standard is empty.\n"
  def load_all_intents(path \\ nil) do
    gold_standard_path =
      Application.app_dir(:brain)
      |> Path.join("priv/evaluation/intent/gold_standard.json")

    case load_from_gold_standard(gold_standard_path) do
      {:ok, examples} when examples != [] ->
        Logger.info("Loaded intent examples from gold standard", %{examples: length(examples)})
        {:ok, examples}

      _ ->
        load_all_intents_legacy(path)
    end
  end

  defp load_from_gold_standard(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            examples =
              Enum.map(data, fn item ->
                %{
                  text: item["text"],
                  intent: item["intent"],
                  tokens: item["tokens"] || [],
                  pos_tags: item["pos_tags"] || [],
                  entities: item["entities"] || []
                }
              end)

            {:ok, examples}

          _ ->
            {:error, :invalid_json}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp load_all_intents_legacy(path) do
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

        Logger.info("Loaded intent examples from legacy directory", %{
          files: length(json_files),
          examples: length(examples)
        })

        {:ok, examples}

      {:error, _reason} ->
        Logger.debug("No legacy intents directory found")
        {:ok, []}
    end
  end

  @doc "Load a single intent file and extract training examples with entity annotations.\n"
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

  @doc "Load custom smalltalk responses.\nReturns a map of action => list of response strings.\n"
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

  @doc "Build a normalized lookup map from entity entries.\nMaps lowercase synonym -> entity info or list of entity infos.\n\nIf an entry has a `types` array, each type becomes a separate entity entry.\nThe code simply reads what's in the data without interpretation.\n"
  def build_entity_lookup(entities) when is_map(entities) do
    Enum.reduce(entities, %{}, fn {_file_entity_type, entries}, acc ->
      Enum.reduce(entries, acc, fn entry, inner_acc ->
        value = entry.value
        synonyms = Map.get(entry, :synonyms, [])

        # A record often lists its value among its own synonyms; each surface
        # form is indexed once per record, or the record would appear twice.
        [value | synonyms]
        |> Enum.uniq_by(&normalize_text/1)
        |> Enum.reduce(inner_acc, fn synonym, lookup ->
          normalized = normalize_text(synonym)

          if String.length(normalized) >= 2 do
            entity_entries = build_entries_from_data(entry, value, synonym)
            add_entries_to_lookup(lookup, normalized, entity_entries)
          else
            lookup
          end
        end)
      end)
    end)
  end

  defp build_entries_from_data(entry, value, original) do
    types = Map.get(entry, :types) || Map.get(entry, "types")

    if is_list(types) and types != [] do
      Enum.map(types, fn type_data ->
        build_entry_from_type_data(type_data, value, original)
      end)
    else
      entity_type = Map.get(entry, :entity_type) || Map.get(entry, "entity_type") || "unknown"
      metadata = Map.get(entry, :metadata) || Map.get(entry, "metadata") || %{}

      [
        %{
          entity_type: entity_type,
          value: value,
          original: original,
          metadata: metadata
        }
      ]
    end
  end

  # The rest of the pipeline reads these fields as atoms (`&1.entity_type`), but
  # a `types` entry decoded from JSON arrives with string keys. Only declared
  # keys are converted; anything else is preserved under `:metadata` rather than
  # dropped or turned into a new atom at runtime.
  @type_entry_keys %{
    "entity_type" => :entity_type,
    "confidence" => :confidence,
    "source" => :source,
    "country" => :country,
    "state" => :state,
    "subcountry" => :subcountry
  }

  defp build_entry_from_type_data(type_data, value, original) do
    {known, extra} =
      Enum.reduce(type_data, {%{}, %{}}, fn {key, val}, {known, extra} ->
        cond do
          is_atom(key) -> {Map.put(known, key, val), extra}
          Map.has_key?(@type_entry_keys, key) -> {Map.put(known, @type_entry_keys[key], val), extra}
          true -> {known, Map.put(extra, key, val)}
        end
      end)

    known
    |> Map.put(:value, value)
    |> Map.put(:original, original)
    |> put_if_present(:metadata, extra)
    |> ensure_entity_type()
  end

  defp ensure_entity_type(entry) do
    entity_type = Map.get(entry, :entity_type) || Map.get(entry, "entity_type")

    if entity_type do
      entry
    else
      Map.put(entry, :entity_type, "unknown")
    end
  end

  defp add_entries_to_lookup(lookup, normalized, entries) when is_list(entries) do
    case Map.get(lookup, normalized) do
      nil ->
        if length(entries) == 1 do
          Map.put(lookup, normalized, hd(entries))
        else
          Map.put(lookup, normalized, entries)
        end

      existing when is_list(existing) ->
        Map.put(lookup, normalized, entries ++ existing)

      existing when is_map(existing) ->
        Map.put(lookup, normalized, entries ++ [existing])
    end
  end

  @doc "Build city lookup from loaded city data.\nMaps lowercase city name -> city info\n"
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

  @doc """
  Whether `word`, as written, is an ordinary English word: the brain's own
  lexicon (`Brain.Lexicon.UserDefined`) records more uses of it as an
  ordinary word (`ordinary_usage` facts) than as any entity type
  (`sense_usage` facts), so that, before anything else is known, the ordinary
  reading is the more probable one. `Brain.Lexicon.Seeder` seeds both from
  the senses SemCor counted and the ordinary ones from the closed-class
  vocabulary too, and they are the counts `Brain.Analysis.EntityTypeScorer`
  weighs a gazetteer match's readings by. With the counts as the seeder
  derives them (2026-10-13), the state codes "in", "or", "me", "hi" and "ok"
  are ordinary words; "dc" (7 uses as a place, 1 ordinary) and "tx" are not.

  `store` is the `Brain.Lexicon.UserDefined` store to read.
  """
  @spec ordinary_word?(String.t(), GenServer.server()) :: boolean()
  def ordinary_word?(word, store \\ Brain.Lexicon.UserDefined) when is_binary(word) do
    uses = fn key ->
      word
      |> String.downcase()
      |> Brain.Lexicon.UserDefined.facts([kind: "property", key: key], store)
      |> Enum.map(fn
        %{value: %{"count" => n}} when is_integer(n) and n >= 0 ->
          n

        fact ->
          raise "DataLoaders: #{fact.key} fact for #{inspect(fact.word)} (#{fact.source}) has no count: #{inspect(fact.value)}"
      end)
      |> Enum.sum()
    end

    uses.("ordinary_usage") > uses.("sense_usage")
  end

  @doc """
  Build US city lookup from loaded US city data.

  Maps each lowercase city name to the city's info, and the name followed by
  its state's name ("springfield illinois", "springfield, illinois") or code
  after a comma ("owosso, mi") too.

  A city's name is a reading of the name like any other, whether or not the
  name is also an ordinary word (Points, Check, Money): which a match is, the
  place or the word, is decided where the match is read
  (`Brain.Analysis.EntityTypeScorer` weighs the word's ordinary uses against
  its uses as a place, its casing, its tag and its context), not here.

  The name followed by the bare state code ("austin tx") is a key only when
  the code is not an ordinary word (`ordinary_word?`, given as a function of
  the code): "rome in" or "salem or" is far more often a place and a word
  than a city and its state, and the gazetteer's keys are lowercase, so how
  the code was written cannot tell them apart.
  """
  def build_us_city_lookup(cities, ordinary_word?) when is_list(cities) and is_function(ordinary_word?, 1) do
    code_is_word =
      cities
      |> Enum.map(&String.downcase(Map.get(&1, :state_code) || ""))
      |> Enum.uniq()
      |> Map.new(&{&1, ordinary_word?.(&1)})

    Enum.reduce(cities, %{}, fn city, acc ->
      name = Map.get(city, :city) || ""
      state_code = Map.get(city, :state_code) || ""
      state_name = Map.get(city, :state_name) || ""
      county = Map.get(city, :county) || ""
      normalized = normalize_text(name)
      code = String.downcase(state_code)

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

        acc =
          acc
          |> Map.put(normalized, city_info)
          |> Map.put("#{normalized}, #{code}", city_info)
          |> Map.put("#{normalized}, #{String.downcase(state_name)}", city_info)
          |> Map.put("#{normalized} #{String.downcase(state_name)}", city_info)

        if Map.fetch!(code_is_word, code), do: acc, else: Map.put(acc, "#{normalized} #{code}", city_info)
      else
        acc
      end
    end)
  end

  @doc "Build artist lookup from loaded artist data.\nMaps lowercase artist name -> artist info\n"
  def build_artist_lookup(artists) when is_list(artists) do
    Enum.reduce(artists, %{}, fn artist, acc ->
      name = Map.get(artist, :artist_name) || ""
      normalized = normalize_text(name)

      if String.length(normalized) >= 2 do
        Map.put(acc, normalized, %{
          entity_type: "music_artist",
          value: name,
          genre: Map.get(artist, :artist_genre),
          country: Map.get(artist, :country)
        })
      else
        acc
      end
    end)
  end

  @doc "Build emoji lookup from loaded emoji data.\nMaps lowercase emoji name -> emoji info\n"
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


  @doc "Load negative training examples from *_negative_en.json files.\n\nNegative examples are phrases that should NOT be classified as a particular intent.\nThey help the model learn to distinguish between similar-sounding but semantically\ndifferent inputs (e.g., \"tell me about the weather\" should NOT be meta.self_knowledge).\n\n## Format\nEach negative example file contains:\n```json\n[\n  {\"text\": \"tell me about the weather\", \"correct_intent\": \"weather.query\"},\n  {\"text\": \"what can you tell me about music\", \"correct_intent\": \"music.search\"}\n]\n```\n\nThe filename indicates what intent these are negative for (e.g., meta.self_knowledge_negative_en.json).\n"
  def load_negative_examples(path \\ nil) do
    negative_dir = path || get_data_path("intents/negative_examples")

    case File.ls(negative_dir) do
      {:ok, files} ->
        negative_files = Enum.filter(files, &String.ends_with?(&1, ".json"))

        examples =
          Enum.flat_map(negative_files, fn file ->
            file_path = Path.join(negative_dir, file)
            negative_for = extract_negative_intent_name(file)

            case load_negative_file(file_path, negative_for) do
              {:ok, file_examples} -> file_examples
              {:error, _} -> []
            end
          end)

        Logger.info("Loaded negative examples", %{
          files: length(negative_files),
          examples: length(examples)
        })

        {:ok, examples}

      {:error, reason} ->
        Logger.warning("Failed to list intents directory for negatives", %{reason: reason})
        {:ok, []}
    end
  end

  @doc "Load a single negative examples file.\n"
  def load_negative_file(path, negative_for) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            examples =
              Enum.map(data, fn item ->
                %{
                  text: Map.get(item, "text", ""),
                  intent: Map.get(item, "correct_intent", "unknown"),
                  entities: [],
                  negative_for: negative_for
                }
              end)
              |> Enum.filter(fn ex -> ex.text != "" end)

            {:ok, examples}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end


  @doc "Build intent label vocabulary from training examples.\n\nReturns `{label_to_idx, idx_to_label}` maps.\n"
  def build_intent_vocabulary(examples) do
    intents =
      examples
      |> Enum.map(fn ex -> ex.intent end)
      |> Enum.uniq()
      |> Enum.sort()

    label_to_idx =
      intents
      |> Enum.with_index()
      |> Enum.into(%{})

    idx_to_label =
      label_to_idx
      |> Enum.map(fn {k, v} -> {v, k} end)
      |> Enum.into(%{})

    Logger.info("Built intent vocabulary", %{num_intents: length(intents)})

    {label_to_idx, idx_to_label}
  end

  @doc "Build BIO tag vocabulary for NER training.\n\nExtracts all entity types from training data and creates BIO tags.\n"
  def build_bio_vocabulary(examples) do
    entity_types =
      examples
      |> Enum.flat_map(fn ex ->
        Enum.map(ex.entities || [], fn e ->
          e[:type] || e["type"] || "unknown"
        end)
      end)
      |> Enum.uniq()
      |> Enum.sort()

    bio_tags =
      ["O"] ++
        Enum.flat_map(entity_types, fn type ->
          ["B-#{type}", "I-#{type}"]
        end)

    bio_to_idx =
      bio_tags
      |> Enum.with_index()
      |> Enum.into(%{})

    idx_to_bio =
      bio_to_idx
      |> Enum.map(fn {k, v} -> {v, k} end)
      |> Enum.into(%{})

    Logger.info("Built BIO vocabulary", %{
      entity_types: length(entity_types),
      bio_tags: length(bio_tags)
    })

    {bio_to_idx, idx_to_bio}
  end

  @doc "Convert tokens to indices using vocabulary.\n\nUnknown tokens are mapped to <UNK> (index 1).\n"
  def tokens_to_indices(tokens, vocab) do
    unk_idx = Map.get(vocab, "<UNK>", 1)
    Enum.map(tokens, fn token -> Map.get(vocab, token, unk_idx) end)
  end

  @doc "Pad or truncate sequence to target length.\n"
  def pad_sequence(indices, target_length, pad_idx \\ 0) do
    current_length = length(indices)

    cond do
      current_length == target_length -> indices
      current_length > target_length -> Enum.take(indices, target_length)
      true -> indices ++ List.duplicate(pad_idx, target_length - current_length)
    end
  end

  defp extract_negative_intent_name(filename) do
    filename
    |> String.replace("_negative_en.json", "")
    |> String.replace("_negative.json", "")
    |> String.replace(" ", ".")
  end

  defp get_data_path(filename), do: Brain.data_path(filename)

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
          Enum.any?(row, fn {_k, v} -> v != "" and v != nil end)
        end)

      _ ->
        []
    end
  end

  defp parse_csv_line(line) do
    parse_csv_fields(line, [], "", false)
  end

  defp parse_csv_fields("", acc, current, _in_quotes) do
    Enum.reverse([String.trim(current) | acc])
  end

  defp parse_csv_fields(<<"\"", rest::binary>>, acc, current, false) do
    parse_csv_fields(rest, acc, current, true)
  end

  defp parse_csv_fields(<<"\"\"", rest::binary>>, acc, current, true) do
    parse_csv_fields(rest, acc, current <> "\"", true)
  end

  defp parse_csv_fields(<<"\"", rest::binary>>, acc, current, true) do
    parse_csv_fields(rest, acc, current, false)
  end

  defp parse_csv_fields(<<",", rest::binary>>, acc, current, false) do
    parse_csv_fields(rest, [String.trim(current) | acc], "", false)
  end

  defp parse_csv_fields(<<char::utf8, rest::binary>>, acc, current, in_quotes) do
    parse_csv_fields(rest, acc, current <> <<char::utf8>>, in_quotes)
  end

  defp extract_entity_type(filename) do
    filename
    |> String.replace("_entries_en.json", "")
    |> String.replace(".json", "")
    |> underscored_name!(filename)
  end

  # Entity types, slot names and context names are written with underscores
  # everywhere; a hyphenated one is data that was not migrated, not a second
  # spelling to convert.
  defp underscored_name!(name, source) do
    if String.contains?(name, "-") do
      raise "DataLoaders: #{inspect(name)} from #{inspect(source)} contains a hyphen; names use underscores"
    end

    name
  end

  # A source entry may declare several readings of the same surface form, as
  # `data/entities/ambiguous_name_location_entries_en.json` does: "Austin" is
  # both a person and a city, each with its own confidence.
  #
  # This used to keep only value/synonyms plus an `entity_type` derived from the
  # *filename*, so every such entry collapsed into one junk type -- all 35
  # ambiguous names loaded as "ambiguous_name_location" rather than as person +
  # city, and the per-type confidence was dropped here, before anything
  # downstream could read it.
  #
  # `build_entries_from_data/3` has always had a `types` branch ready for this;
  # it was simply never reached, because this function stripped its input.
  # Measured 2026-09-23.
  defp parse_entity_data(data, entity_type) when is_list(data) do
    data
    |> Enum.map(fn item ->
      base = %{
        value: Map.get(item, "value") || Map.get(item, "name") || "",
        synonyms: item |> Map.get("synonyms", []) |> List.wrap(),
        entity_type: entity_type
      }

      base
      |> put_if_present(:types, Map.get(item, "types"))
      |> put_if_present(:metadata, Map.get(item, "metadata"))
    end)
    |> Enum.filter(fn entry -> entry.value != "" end)
  end

  defp parse_entity_data(%{"entries" => entries}, entity_type) when is_list(entries) do
    parse_entity_data(entries, entity_type)
  end

  defp parse_entity_data(_, _entity_type) do
    []
  end

  defp put_if_present(map, _key, nil), do: map
  defp put_if_present(map, _key, []), do: map
  defp put_if_present(map, _key, empty) when empty == %{}, do: map
  defp put_if_present(map, key, value), do: Map.put(map, key, value)

  # An intent file is named by its intent. A name with " - " is a contextual
  # variant that was not collapsed into the intent its action names. Spaces occur
  # only in Dialogflow display-named files such as "Default Fallback Intent".
  defp extract_intent_name(filename) do
    name =
      filename
      |> String.replace("_usersays_en.json", "")
      |> String.replace("_usersays.json", "")
      |> String.replace(".json", "")

    if String.contains?(name, " - ") do
      raise "DataLoaders: #{inspect(filename)} names a contextual variant; collapse it into the intent its action names"
    end

    String.replace(name, " ", ".")
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

  defp parse_intent_data(_, _intent_name) do
    []
  end

  defp extract_example_with_entities(example) do
    case Map.get(example, "data") do
      nil ->
        text = Map.get(example, "text", "")
        {:ok, text, []}

      data when is_list(data) ->
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
    |> underscored_name!(meta)
  end

  defp normalize_text(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_text(_) do
    ""
  end
end
