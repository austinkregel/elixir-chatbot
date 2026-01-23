defmodule ChatBot.ML.EntityExtractor do
  @moduledoc """
  Entity extraction using gazetteer lookups and classical NLP techniques.

  This module extracts entities from user input text using:
  - Gazetteer lookups for known entities (cities, artists, devices, etc.)
  - BIO-tagged sequence model for unknown entity detection
  - Token-based pattern matching for system entities (dates, numbers)

  Avoids regex in favor of tokenizer-based approaches.
  """

  require Logger

  alias ChatBot.ML.{Gazetteer, Tokenizer, EntityTrainer}

  @type entity_match :: %{
          entity: String.t(),
          value: String.t(),
          match: String.t(),
          start_pos: integer(),
          end_pos: integer(),
          confidence: float()
        }

  @type entity_map :: %{String.t() => %{entity: String.t(), value: String.t()}}

  # Common words to ignore for location detection
  @common_words MapSet.new([
                  "i",
                  "the",
                  "a",
                  "an",
                  "and",
                  "but",
                  "or",
                  "so",
                  "if",
                  "when",
                  "where",
                  "what",
                  "how",
                  "why",
                  "who",
                  "which",
                  "that",
                  "this",
                  "these",
                  "those",
                  "hello",
                  "hi",
                  "hey",
                  "thanks",
                  "thank",
                  "please",
                  "yes",
                  "no",
                  "yeah",
                  "nope",
                  "ok",
                  "okay",
                  "sure",
                  "can",
                  "could",
                  "would",
                  "should",
                  "do",
                  "does",
                  "did",
                  "is",
                  "are",
                  "was",
                  "were",
                  "have",
                  "has",
                  "had",
                  "will",
                  "shall",
                  "may",
                  "might",
                  "must",
                  "let",
                  "me",
                  "my",
                  "you",
                  "your",
                  "we",
                  "our",
                  "they",
                  "their",
                  "it",
                  "its",
                  "he",
                  "she",
                  "him",
                  "her",
                  "am",
                  "be",
                  "been",
                  "being",
                  "get",
                  "got",
                  "give",
                  "gave",
                  "go",
                  "going",
                  "gone",
                  "come",
                  "came",
                  "tell",
                  "told",
                  "say",
                  "said",
                  "ask",
                  "asked",
                  "know",
                  "knew",
                  "think",
                  "thought",
                  "want",
                  "wanted",
                  "need",
                  "needed",
                  "make",
                  "made",
                  "see",
                  "saw",
                  "look",
                  "looked",
                  "find",
                  "found",
                  "take",
                  "took",
                  "put",
                  "just",
                  "also",
                  "too",
                  "very",
                  "really",
                  "actually",
                  "probably",
                  "maybe",
                  "perhaps",
                  "about",
                  "for",
                  "with",
                  "from",
                  "into",
                  "after",
                  "before",
                  "during",
                  "until",
                  "since",
                  "not",
                  "don't",
                  "doesn't",
                  "didn't",
                  "won't",
                  "wouldn't",
                  "can't",
                  "couldn't",
                  "shouldn't",
                  "there",
                  "here",
                  "now",
                  "then",
                  "today",
                  "tomorrow",
                  "yesterday",
                  "play",
                  "show",
                  "turn",
                  "set",
                  "check",
                  "on",
                  "off",
                  "up",
                  "down"
                ])

  # Prepositions that often precede locations
  @location_prepositions MapSet.new(["in", "at", "near", "around", "from", "to"])

  # Relative date words
  @relative_dates MapSet.new(["today", "tomorrow", "yesterday"])

  # Day names
  @day_names MapSet.new([
               "monday",
               "tuesday",
               "wednesday",
               "thursday",
               "friday",
               "saturday",
               "sunday"
             ])

  # Month names
  @month_names MapSet.new([
                 "january",
                 "february",
                 "march",
                 "april",
                 "may",
                 "june",
                 "july",
                 "august",
                 "september",
                 "october",
                 "november",
                 "december",
                 "jan",
                 "feb",
                 "mar",
                 "apr",
                 "jun",
                 "jul",
                 "aug",
                 "sep",
                 "oct",
                 "nov",
                 "dec"
               ])

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Load entity maps from saved gazetteer or build fresh.
  Returns {:ok, maps} or {:error, reason}.
  """
  def load_entity_maps do
    # Try to load from saved gazetteer file
    models_path = Application.get_env(:chat_bot, :ml)[:models_path]
    gazetteer_path = Path.join(models_path, "gazetteer.term")

    case File.read(gazetteer_path) do
      {:ok, binary} ->
        try do
          entity_maps = :erlang.binary_to_term(binary)

          # Store in Agent for fast access
          case start_agent(entity_maps) do
            {:ok, _} -> {:ok, entity_maps}
            {:error, reason} -> {:error, reason}
          end
        rescue
          e ->
            Logger.warning("Failed to load gazetteer, falling back to legacy", %{
              error: inspect(e)
            })

            load_entity_maps_legacy()
        end

      {:error, _} ->
        # Fall back to legacy loading
        load_entity_maps_legacy()
    end
  end

  defp start_agent(entity_maps) do
    case Agent.start_link(fn -> entity_maps end, name: __MODULE__) do
      {:ok, pid} ->
        Logger.info("Entity maps loaded successfully", %{entities_count: map_size(entity_maps)})
        {:ok, pid}

      {:error, {:already_started, _pid}} ->
        # Update existing agent
        Agent.update(__MODULE__, fn _ -> entity_maps end)
        {:ok, :updated}

      {:error, reason} ->
        Logger.error("Failed to start Agent", %{reason: reason})
        {:error, reason}
    end
  end

  @doc """
  Legacy entity loading (for backward compatibility).
  """
  def load_entity_maps_legacy do
    base_path = Application.get_env(:chat_bot, :ml)[:training_data_path]
    entities_dir = Path.join(base_path, "entities")

    dir_maps =
      case File.ls(entities_dir) do
        {:ok, files} ->
          Enum.reduce(files, %{}, fn file, acc ->
            if String.ends_with?(file, ".json") do
              file_path = Path.join(entities_dir, file)

              case File.read(file_path) do
                {:ok, content} ->
                  case Jason.decode(content) do
                    {:ok, %{"entries" => entries}} when is_list(entries) ->
                      merge_entries_map(acc, base_name(file), entries)

                    {:ok, list} when is_list(list) ->
                      merge_entries_map(acc, base_name(file), list)

                    _ ->
                      acc
                  end

                _ ->
                  acc
              end
            else
              acc
            end
          end)

        _ ->
          %{}
      end

    case start_agent(dir_maps) do
      {:ok, _} -> {:ok, dir_maps}
      {:error, reason} -> {:error, reason}
    end
  end

  defp merge_entries_map(acc, entity_name, entries) do
    Enum.reduce(entries, acc, fn entry, acc2 ->
      value = Map.get(entry, "value") || Map.get(entry, "name")
      synonyms = Map.get(entry, "synonyms", [])
      all_terms = [value | synonyms] |> Enum.reject(&is_nil/1)

      Enum.reduce(all_terms, acc2, fn term, acc3 ->
        normalized = String.downcase(String.trim(term))

        if String.length(normalized) >= 2 do
          # Use entity_type key for consistency with DataLoaders format
          Map.put(acc3, normalized, %{
            entity_type: normalize_entity_name(entity_name),
            value: value
          })
        else
          acc3
        end
      end)
    end)
  end

  defp normalize_entity_name(name) do
    # Remove _entries_en suffix and normalize
    name
    |> String.replace("_entries_en", "")
    |> String.replace("_", "-")
  end

  defp base_name(file), do: file |> String.replace_suffix(".json", "")

  @doc """
  Get entity maps from Agent or load fresh.
  """
  def get_entity_maps do
    try do
      Agent.get(__MODULE__, & &1)
    rescue
      _e ->
        load_entity_maps_fallback()
    catch
      :exit, _ ->
        load_entity_maps_fallback()
    end
  end

  defp load_entity_maps_fallback do
    case load_entity_maps() do
      {:ok, maps} -> maps
      {:error, _} -> %{}
    end
  end

  @doc """
  Extract entities from text using gazetteer lookups and pattern matching.
  Returns a list of entity matches with positions and confidence scores.
  """
  def extract_entities(text, entity_maps \\ nil) do
    entity_maps = entity_maps || get_entity_maps()

    # Tokenize the text
    tokens = Tokenizer.tokenize(text)

    # Extract entities from gazetteer
    gazetteer_entities = extract_gazetteer_entities(tokens, entity_maps)

    # Extract system entities (dates, numbers)
    system_entities = extract_system_entities(tokens, text)

    # Extract location hints from context
    location_entities = extract_location_hints(tokens, entity_maps)

    # Combine and resolve conflicts
    all_entities = gazetteer_entities ++ system_entities ++ location_entities

    resolve_entity_conflicts(all_entities)
  end

  @doc """
  Extract entities using the BIO-tagged model (if available).
  Falls back to gazetteer if model not loaded.
  """
  def extract_entities_with_model(text, opts \\ []) do
    entity_maps = Keyword.get(opts, :entity_maps) || get_entity_maps()

    # First try gazetteer-based extraction
    gazetteer_entities = extract_entities(text, entity_maps)

    # Then try BIO model for additional entities
    model_entities =
      case EntityTrainer.load_model() do
        {:ok, model} ->
          tokens = Tokenizer.tokenize_words(text)
          predictions = EntityTrainer.predict(tokens, model)

          EntityTrainer.extract_entities_from_bio(predictions)
          |> Enum.map(fn entity ->
            %{
              entity: entity.entity,
              value: entity.value,
              match: entity.value,
              # Position not tracked in BIO
              start_pos: 0,
              end_pos: String.length(entity.value) - 1,
              confidence: entity.confidence
            }
          end)

        {:error, _} ->
          []
      end

    # Merge and deduplicate
    all_entities = gazetteer_entities ++ model_entities

    all_entities
    |> Enum.uniq_by(fn e -> {String.downcase(e.value), e.entity} end)
    |> resolve_entity_conflicts()
  end

  # ============================================================================
  # Gazetteer-based Entity Extraction
  # ============================================================================

  defp extract_gazetteer_entities(tokens, entity_maps) do
    # Try to use the Gazetteer GenServer if available
    if Gazetteer.loaded?() do
      extract_with_gazetteer_server(tokens)
    else
      extract_with_local_maps(tokens, entity_maps)
    end
  end

  defp extract_with_gazetteer_server(tokens) do
    # Normalize tokens - lowercase and strip punctuation for lookup
    token_texts =
      Enum.map(tokens, fn t ->
        t.text
        |> String.downcase()
        |> String.replace(~r/[^\w\s-]/u, "")
        |> String.trim()
      end)

    Gazetteer.lookup_spans(token_texts)
    |> Enum.map(fn {start_idx, end_idx, entity_info} ->
      start_token = Enum.at(tokens, start_idx)
      end_token = Enum.at(tokens, end_idx)

      matched_tokens = Enum.slice(tokens, start_idx..end_idx)
      match_text = Enum.map(matched_tokens, & &1.text) |> Enum.join(" ")

      # Handle both :entity_type and :entity keys for backwards compatibility
      entity_type = Map.get(entity_info, :entity_type) || Map.get(entity_info, :entity, "unknown")
      entity_value = Map.get(entity_info, :value, match_text)

      %{
        entity: entity_type,
        value: entity_value,
        match: match_text,
        start_pos: start_token.start_pos,
        end_pos: end_token.end_pos,
        confidence: calculate_confidence(match_text, entity_type)
      }
    end)
  end

  defp extract_with_local_maps(tokens, entity_maps) do
    # Try different span lengths (longest first)
    max_span = 5
    token_count = length(tokens)

    find_all_local_spans(tokens, entity_maps, 0, token_count, max_span, [])
  end

  defp find_all_local_spans(_tokens, _entity_maps, start_idx, token_count, _max_span, acc)
       when start_idx >= token_count do
    resolve_entity_conflicts(Enum.reverse(acc))
  end

  defp find_all_local_spans(tokens, entity_maps, start_idx, token_count, max_span, acc) do
    # Try longest spans first
    match = find_longest_local_match(tokens, entity_maps, start_idx, max_span)

    case match do
      {:ok, end_idx, entity_info, match_text} ->
        start_token = Enum.at(tokens, start_idx)
        end_token = Enum.at(tokens, end_idx)

        # Handle both :entity_type and :entity keys for backwards compatibility
        entity_type =
          Map.get(entity_info, :entity_type) || Map.get(entity_info, :entity, "unknown")

        entity_value = Map.get(entity_info, :value, match_text)

        entity = %{
          entity: entity_type,
          value: entity_value,
          match: match_text,
          start_pos: start_token.start_pos,
          end_pos: end_token.end_pos,
          confidence: calculate_confidence(match_text, entity_type)
        }

        find_all_local_spans(tokens, entity_maps, start_idx + 1, token_count, max_span, [
          entity | acc
        ])

      :not_found ->
        find_all_local_spans(tokens, entity_maps, start_idx + 1, token_count, max_span, acc)
    end
  end

  defp find_longest_local_match(tokens, entity_maps, start_idx, max_span) do
    token_count = length(tokens)
    actual_max = min(max_span, token_count - start_idx)

    if actual_max < 1 do
      :not_found
    else
      actual_max..1//-1
      |> Enum.reduce_while(:not_found, fn span_len, _acc ->
        span_tokens = Enum.slice(tokens, start_idx, span_len)
        phrase = Enum.map(span_tokens, & &1.text) |> Enum.join(" ")
        normalized = String.downcase(phrase)

        case Map.get(entity_maps, normalized) do
          nil ->
            {:cont, :not_found}

          entity_info ->
            {:halt, {:ok, start_idx + span_len - 1, entity_info, phrase}}
        end
      end)
    end
  end

  # ============================================================================
  # System Entity Extraction (No Regex)
  # ============================================================================

  defp extract_system_entities(tokens, _text) do
    number_entities = extract_numbers_from_tokens(tokens)
    date_entities = extract_dates_from_tokens(tokens)

    number_entities ++ date_entities
  end

  defp extract_numbers_from_tokens(tokens) do
    tokens
    |> Enum.filter(fn token -> token.type == :number end)
    |> Enum.map(fn token ->
      %{
        entity: "number",
        value: token.text,
        match: token.text,
        start_pos: token.start_pos,
        end_pos: token.end_pos,
        confidence: 0.9
      }
    end)
  end

  defp extract_dates_from_tokens(tokens) do
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      lower = String.downcase(token.text)

      cond do
        MapSet.member?(@relative_dates, lower) ->
          [
            %{
              entity: "relative_date",
              value: token.text,
              match: token.text,
              start_pos: token.start_pos,
              end_pos: token.end_pos,
              confidence: 0.9
            }
          ]

        MapSet.member?(@day_names, lower) ->
          [
            %{
              entity: "day_name",
              value: token.text,
              match: token.text,
              start_pos: token.start_pos,
              end_pos: token.end_pos,
              confidence: 0.85
            }
          ]

        MapSet.member?(@month_names, lower) ->
          # Check if followed by a number (day)
          maybe_date = check_for_date_pattern(tokens, idx)

          case maybe_date do
            nil ->
              [
                %{
                  entity: "month_name",
                  value: token.text,
                  match: token.text,
                  start_pos: token.start_pos,
                  end_pos: token.end_pos,
                  confidence: 0.8
                }
              ]

            date_entity ->
              [date_entity]
          end

        true ->
          []
      end
    end)
  end

  defp check_for_date_pattern(tokens, month_idx) do
    # Look for patterns like "January 15" or "January 15, 2024"
    next_token = Enum.at(tokens, month_idx + 1)
    month_token = Enum.at(tokens, month_idx)

    if next_token != nil and next_token.type == :number do
      day_num = next_token.text

      # Check for year
      year_token = Enum.at(tokens, month_idx + 2)

      if year_token != nil and year_token.type == :number and String.length(year_token.text) == 4 do
        # Full date: "January 15 2024"
        match_text = "#{month_token.text} #{day_num} #{year_token.text}"

        %{
          entity: "date",
          value: match_text,
          match: match_text,
          start_pos: month_token.start_pos,
          end_pos: year_token.end_pos,
          confidence: 0.9
        }
      else
        # Partial date: "January 15"
        match_text = "#{month_token.text} #{day_num}"

        %{
          entity: "date",
          value: match_text,
          match: match_text,
          start_pos: month_token.start_pos,
          end_pos: next_token.end_pos,
          confidence: 0.85
        }
      end
    else
      nil
    end
  end

  # ============================================================================
  # Location Hint Extraction
  # ============================================================================

  defp extract_location_hints(tokens, entity_maps) do
    # Look for patterns like "in [Capitalized Words]"
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      lower = String.downcase(token.text)

      if MapSet.member?(@location_prepositions, lower) do
        # Look at following tokens for potential location
        extract_following_location(tokens, idx + 1, entity_maps)
      else
        []
      end
    end)
  end

  defp extract_following_location(tokens, start_idx, entity_maps) do
    # Collect capitalized non-common words
    remaining = Enum.drop(tokens, start_idx)

    location_tokens =
      remaining
      |> Enum.take_while(fn token ->
        capitalized?(token.text) and not common_word?(token.text)
      end)

    if length(location_tokens) > 0 do
      location_text = Enum.map(location_tokens, & &1.text) |> Enum.join(" ")
      normalized = String.downcase(location_text)

      # Check if this is already in our gazetteer
      unless Map.has_key?(entity_maps, normalized) do
        first_token = List.first(location_tokens)
        last_token = List.last(location_tokens)

        [
          %{
            entity: "location",
            value: location_text,
            match: location_text,
            start_pos: first_token.start_pos,
            end_pos: last_token.end_pos,
            # Lower confidence for inferred locations
            confidence: 0.7
          }
        ]
      else
        # Already in gazetteer, will be found there
        []
      end
    else
      []
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp calculate_confidence(match_text, entity_type) do
    # Base confidence on match length and entity type
    base = min(0.9, 0.5 + String.length(match_text) * 0.03)

    # Adjust based on entity type
    type_bonus =
      case entity_type do
        "device" -> 0.1
        "room" -> 0.1
        "location" -> 0.05
        "music-artist" -> 0.05
        "music_artist" -> 0.05
        _ -> 0.0
      end

    min(0.95, base + type_bonus)
  end

  defp capitalized?(text) do
    first = String.first(text)
    first != nil and first == String.upcase(first) and first != String.downcase(first)
  end

  defp common_word?(text) do
    MapSet.member?(@common_words, String.downcase(text))
  end

  defp resolve_entity_conflicts(matches) do
    # Sort by start position, then by length (longest first)
    sorted = Enum.sort_by(matches, fn m -> {m.start_pos, -String.length(m.match)} end)

    # Remove overlapping matches, keeping the longest/first
    resolve_overlaps(sorted, [])
  end

  defp resolve_overlaps([], resolved), do: Enum.reverse(resolved)

  defp resolve_overlaps([current | rest], resolved) do
    case resolved do
      [] ->
        resolve_overlaps(rest, [current])

      [last | _] = resolved_list ->
        if current.start_pos <= last.end_pos do
          # Overlap detected, keep the longer match
          if String.length(current.match) > String.length(last.match) do
            resolve_overlaps(rest, [current | tl(resolved_list)])
          else
            resolve_overlaps(rest, resolved_list)
          end
        else
          resolve_overlaps(rest, [current | resolved_list])
        end
    end
  end
end
