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

  alias ChatBot.ML.{Gazetteer, Tokenizer, EntityTrainer, POSTagger}
  alias ChatBot.Analysis.{EntityDisambiguator, IntentRegistry}

  @type entity_match :: %{
          entity: String.t(),
          value: String.t(),
          match: String.t(),
          start_pos: integer(),
          end_pos: integer(),
          confidence: float()
        }

  @type entity_map :: %{String.t() => %{entity_type: String.t(), value: String.t()}}

  # ============================================================================
  # Data-Driven Entity Detection
  # All entity patterns are learned from training data via the Gazetteer
  # No hardcoded word lists - everything uses trained models or Gazetteer lookups
  # ============================================================================

  defp is_location_preposition?(text) do
    # Use POS tagger to detect prepositions (ADP)
    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict([text], model)

        case predictions do
          [{_word, "ADP"}] -> true
          _ -> false
        end

      {:error, _} ->
        false
    end
  end

  defp common_word?(text) do
    # Use POS tagger - common/function words are typically DET, ADP, CONJ, PART
    lower = String.downcase(text)

    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict([lower], model)

        case predictions do
          [{_word, tag}] when tag in ["DET", "ADP", "CONJ", "PART", "PUNCT"] -> true
          _ -> false
        end

      {:error, _} ->
        # Can't determine, assume not common
        false
    end
  end

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

  ## Options

  - `:entity_maps` - Pre-loaded entity maps (optional)
  - `:discourse` - Discourse analysis result for disambiguation context
  - `:speech_act` - Speech act classification result for disambiguation context
  - `:skip_disambiguation` - If true, skip the disambiguation step (default: false)
  """
  def extract_entities(text, opts \\ [])

  def extract_entities(text, opts) when is_list(opts) do
    entity_maps = Keyword.get(opts, :entity_maps) || get_entity_maps()
    discourse = Keyword.get(opts, :discourse)
    speech_act = Keyword.get(opts, :speech_act)
    skip_disambiguation = Keyword.get(opts, :skip_disambiguation, false)

    # Tokenize the text
    tokens = Tokenizer.tokenize(text)

    # Extract entities from gazetteer (may return multiple types per entity)
    gazetteer_entities = extract_gazetteer_entities(tokens, entity_maps)

    # Extract system entities (dates, numbers)
    system_entities = extract_system_entities(tokens, text)

    # Extract location hints from context
    location_entities = extract_location_hints(tokens, entity_maps)

    # Combine and resolve conflicts
    all_entities = gazetteer_entities ++ system_entities ++ location_entities
    resolved_entities = resolve_entity_conflicts(all_entities)

    # Disambiguate entities with multiple types if context is available
    disambiguated_entities =
      if skip_disambiguation or (is_nil(discourse) and is_nil(speech_act)) do
        resolved_entities
      else
        # Pass original text for text-based pattern matching (e.g., introduction detection)
        disambiguate_entities(resolved_entities, tokens, discourse, speech_act, text)
      end

    # Filter out entities below confidence threshold
    min_confidence = Keyword.get(opts, :min_confidence) || get_min_confidence_threshold()
    filter_by_confidence(disambiguated_entities, min_confidence)
  end

  # Legacy support: entity_maps passed directly
  def extract_entities(text, entity_maps) when is_map(entity_maps) do
    extract_entities(text, entity_maps: entity_maps)
  end

  def extract_entities(text, nil) do
    extract_entities(text, [])
  end

  # Get minimum confidence threshold from application config
  defp get_min_confidence_threshold do
    Application.get_env(:chat_bot, :ml)[:entity_confidence_threshold] ||
      Application.get_env(:chat_bot, :ml)[:confidence_threshold] ||
      0.51
  end

  # Filter entities by confidence threshold
  defp filter_by_confidence(entities, min_confidence) when is_float(min_confidence) do
    Enum.filter(entities, fn entity ->
      confidence = Map.get(entity, :confidence) || Map.get(entity, "confidence", 0.0)
      confidence >= min_confidence
    end)
  end

  defp filter_by_confidence(entities, _), do: entities

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
              entity_type: entity.entity_type,
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
    |> Enum.uniq_by(fn e -> {String.downcase(e.value), e.entity_type} end)
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
        |> strip_non_word_chars()
        |> String.trim()
      end)

    Gazetteer.lookup_spans(token_texts)
    |> Enum.map(fn {start_idx, end_idx, entity_info} ->
      start_token = Enum.at(tokens, start_idx)
      end_token = Enum.at(tokens, end_idx)

      matched_tokens = Enum.slice(tokens, start_idx..end_idx)
      match_text = Enum.map(matched_tokens, & &1.text) |> Enum.join(" ")

      # Handle single entity_info or list of possible types
      case entity_info do
        infos when is_list(infos) and length(infos) > 1 ->
          # Multiple possible entity types - keep all for disambiguation
          primary_info = hd(infos)
          primary_type = Map.get(primary_info, :entity_type, "unknown")
          entity_value = Map.get(primary_info, :value, match_text)

          %{
            entity_type: primary_type,
            value: entity_value,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: calculate_confidence(match_text, primary_type, entity_value),
            types: infos
          }

        [single_info] ->
          # List with single entry
          entity_type = Map.get(single_info, :entity_type, "unknown")
          entity_value = Map.get(single_info, :value, match_text)

          %{
            entity_type: entity_type,
            value: entity_value,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: calculate_confidence(match_text, entity_type, entity_value)
          }

        single_info when is_map(single_info) ->
          # Single entity info
          entity_type = Map.get(single_info, :entity_type, "unknown")
          entity_value = Map.get(single_info, :value, match_text)

          %{
            entity_type: entity_type,
            value: entity_value,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: calculate_confidence(match_text, entity_type, entity_value)
          }

        _ ->
          # Unknown format
          %{
            entity_type: "unknown",
            value: match_text,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: 0.5
          }
      end
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
      {:ok, end_idx, entity_infos, match_text} ->
        start_token = Enum.at(tokens, start_idx)
        end_token = Enum.at(tokens, end_idx)

        # Create an entity for each type (entity_infos is now always a list)
        new_entities =
          Enum.map(entity_infos, fn entity_info ->
            entity_type = Map.get(entity_info, :entity_type, "unknown")
            entity_value = Map.get(entity_info, :value, match_text)

            %{
              entity_type: entity_type,
              value: entity_value,
              match: match_text,
              start_pos: start_token.start_pos,
              end_pos: end_token.end_pos,
              confidence: calculate_confidence(match_text, entity_type, entity_value)
            }
          end)

        find_all_local_spans(tokens, entity_maps, start_idx + 1, token_count, max_span, new_entities ++ acc)

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

          entity_info when is_list(entity_info) ->
            # Multiple entity types for this entry - return all of them
            {:halt, {:ok, start_idx + span_len - 1, entity_info, phrase}}

          entity_info when is_map(entity_info) ->
            # Single entity type - wrap in list for consistent handling
            {:halt, {:ok, start_idx + span_len - 1, [entity_info], phrase}}
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
        entity_type: "number",
        value: token.text,
        match: token.text,
        start_pos: token.start_pos,
        end_pos: token.end_pos,
        confidence: 0.9
      }
    end)
  end

  defp extract_dates_from_tokens(tokens) do
    # Use Gazetteer to look up temporal entities - no hardcoded word lists
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      lower = String.downcase(token.text)

      # Look up in Gazetteer for temporal entities
      case Gazetteer.lookup(lower) do
        :not_found ->
          []

        {:ok, result} ->
          matches = if is_map(result), do: [result], else: result

          # Check if any match is a temporal type
          temporal_match =
            Enum.find(matches, fn match ->
              entity_type = match[:entity_type] || match["entity_type"] || ""

              entity_type in ~w(
                sys-date date relative_date day weekday month
                day_name month_name time sys-time
              )
            end)

          case temporal_match do
            nil ->
              []

            match ->
              entity_type = match[:entity_type] || match["entity_type"]

              # For month names, check if followed by a number (date pattern)
              if entity_type in ["month", "month_name"] do
                maybe_date = check_for_date_pattern(tokens, idx)

                case maybe_date do
                  nil ->
                    [
                      %{
                        entity_type: entity_type,
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
              else
                [
                  %{
                    entity_type: entity_type,
                    value: token.text,
                    match: token.text,
                    start_pos: token.start_pos,
                    end_pos: token.end_pos,
                    confidence: 0.9
                  }
                ]
              end
          end

        _ ->
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
          entity_type: "date",
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
          entity_type: "date",
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
    # Use POS tagger to detect prepositions (ADP) rather than keyword list
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      if is_location_preposition?(token.text) do
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
            entity_type: "location",
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

  defp calculate_confidence(match_text, entity_type, entity_value) do
    # Base confidence on match length and entity type
    base = min(0.9, 0.5 + String.length(match_text) * 0.03)

    # Adjust based on entity type
    type_bonus =
      case entity_type do
        "device" -> 0.1
        "room" -> 0.1
        "person" -> 0.1
        "location" -> 0.05
        "music-artist" -> 0.05
        "music_artist" -> 0.05
        _ -> 0.0
      end

    # Reduce confidence if casing doesn't match (e.g., "friend" vs "Friend")
    # This helps avoid false matches like "friend" being tagged as location "Friend"
    # Uses general linguistic pattern: capitalized entity value (proper noun) vs lowercase match (common word)
    casing_penalty =
      if entity_value && match_text != entity_value do
        # Check if it's a casing mismatch (same when lowercased)
        match_lower = String.downcase(match_text)
        value_lower = String.downcase(entity_value)

        if match_lower == value_lower && match_text != entity_value do
          # Determine penalty based on linguistic pattern:
          # - Stronger penalty when entity value is capitalized (proper noun) and match is lowercase (common word)
          # - This indicates a likely false positive (e.g., "friend" matching "Friend" location)
          value_is_capitalized = capitalized?(entity_value)
          match_is_lowercase = match_text == match_lower

          penalty =
            if value_is_capitalized && match_is_lowercase do
              # Proper noun entity matched by lowercase common word - high penalty
              0.3
            else
              # Other casing mismatches - moderate penalty
              0.15
            end

          -penalty
        else
          0.0
        end
      else
        0.0
      end

    min(0.95, max(0.1, base + type_bonus + casing_penalty))
  end

  defp capitalized?(text) do
    first = String.first(text)
    first != nil and first == String.upcase(first) and first != String.downcase(first)
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

  # Strip non-word characters (keeping letters, digits, spaces, and hyphens)
  # Unicode-aware replacement for regex: ~r/[^\w\s-]/u
  defp strip_non_word_chars(text) do
    text
    |> String.graphemes()
    |> Enum.filter(&word_or_space_or_hyphen?/1)
    |> Enum.join()
  end

  defp word_or_space_or_hyphen?(grapheme) do
    case grapheme do
      "-" -> true
      " " -> true
      "\t" -> true
      "\n" -> true
      <<c::utf8>> when c in ?a..?z or c in ?A..?Z or c in ?0..?9 -> true
      <<c::utf8>> when c > 127 -> letter_codepoint?(c)
      _ -> false
    end
  end

  defp letter_codepoint?(codepoint) do
    # Check if codepoint is a Unicode letter (basic check for common ranges)
    # Latin Extended, Greek, Cyrillic, etc.
    # Latin Extended
    # Greek
    # Cyrillic
    # Latin Extended Additional
    (codepoint >= 0x00C0 and codepoint <= 0x024F) or
      (codepoint >= 0x0370 and codepoint <= 0x03FF) or
      (codepoint >= 0x0400 and codepoint <= 0x04FF) or
      (codepoint >= 0x1E00 and codepoint <= 0x1EFF)
  end

  # ============================================================================
  # Entity Disambiguation
  # ============================================================================

  defp disambiguate_entities(entities, tokens, discourse, speech_act, original_text) do
    # Extract classified intent from speech_act indicators
    classified_intent = extract_intent_from_speech_act(speech_act)

    # Build context for disambiguation
    # Include original text for text-based pattern matching (e.g., "I'm Austin" detection)
    context = %{
      discourse: discourse,
      speech_act: speech_act,
      intent: classified_intent,
      original_text: original_text
    }

    # Try to get POS tags for better disambiguation
    pos_tagged = get_pos_tags(tokens)

    # Use the EntityDisambiguator to resolve entities with multiple types
    # or entities with a single ambiguous type that needs inference
    disambiguated =
      entities
      |> Enum.map(fn entity ->
        # Check if this entity has multiple possible types
        types = get_entity_types(entity)
        entity_type = entity[:entity_type] || ""

        # Disambiguate if: multiple types OR single ambiguous type needing inference
        needs_disambiguation =
          length(types) > 1 or EntityDisambiguator.requires_inference?(entity_type)

        if needs_disambiguation do
          # Disambiguate this entity
          result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

          # Emit telemetry for disambiguation
          :telemetry.execute(
            [:chat_bot, :analysis, :disambiguation, :entity],
            %{
              type_count: max(length(types), 1),
              selected_type: result[:entity_type]
            },
            %{
              value: entity[:value],
              available_types: Enum.map(types, &(&1[:entity_type] || &1[:type])),
              context_type: context_type(context),
              pos_pattern: extract_pos_pattern(pos_tagged)
            }
          )

          result
        else
          entity
        end
      end)

    # Emit summary telemetry
    ambiguous_count =
      Enum.count(entities, fn e ->
        types = get_entity_types(e)
        entity_type = e[:entity_type] || ""
        length(types) > 1 or EntityDisambiguator.requires_inference?(entity_type)
      end)

    if ambiguous_count > 0 do
      :telemetry.execute(
        [:chat_bot, :analysis, :disambiguation, :complete],
        %{
          total_entities: length(entities),
          ambiguous_entities: ambiguous_count
        },
        %{
          has_discourse: discourse != nil,
          has_speech_act: speech_act != nil
        }
      )
    end

    disambiguated
  end

  defp context_type(context) do
    # Use the classified intent from context, or extract from speech_act
    intent = context[:intent] || extract_intent_from_speech_act(context[:speech_act])

    cond do
      # Use IntentRegistry predicates for all domain detection
      IntentRegistry.introduction_intent?(intent) -> :introduction
      IntentRegistry.weather_intent?(intent) -> :weather
      IntentRegistry.music_intent?(intent) -> :music
      IntentRegistry.device_intent?(intent) -> :device
      true -> :default
    end
  end

  defp extract_intent_from_speech_act(nil), do: nil

  defp extract_intent_from_speech_act(speech_act) do
    indicators = get_field(speech_act, :indicators) || []

    Enum.find_value(indicators, fn indicator ->
      case String.split(to_string(indicator), ":", parts: 2) do
        ["intent", intent] -> intent
        _ -> nil
      end
    end)
  end

  # Helper to safely get a field from either a struct or map
  defp get_field(nil, _key), do: nil
  defp get_field(struct, key) when is_struct(struct), do: Map.get(struct, key)
  defp get_field(map, key) when is_map(map), do: Map.get(map, key)
  defp get_field(_, _), do: nil

  defp extract_pos_pattern(pos_tagged) do
    pos_tagged
    |> Enum.take(3)
    |> Enum.map(fn
      {_token, tag} -> tag
      tag -> tag
    end)
    |> Enum.join("-")
  end

  defp get_pos_tags(tokens) do
    # Try to use POS tagger if model is available
    case POSTagger.load_model() do
      {:ok, model} ->
        # Extract just the text from tokens
        token_texts =
          Enum.map(tokens, fn
            %{text: text} -> text
            text when is_binary(text) -> text
            _ -> ""
          end)

        POSTagger.predict(token_texts, model)

      {:error, _} ->
        # No POS model available, return tokens without tags
        Enum.map(tokens, fn
          %{text: text} -> {text, "X"}
          text when is_binary(text) -> {text, "X"}
          _ -> {"", "X"}
        end)
    end
  end

  defp get_entity_types(entity) do
    cond do
      is_list(Map.get(entity, :types)) ->
        entity.types

      is_list(Map.get(entity, "types")) ->
        entity["types"]

      # Entity already has a single type
      true ->
        []
    end
  end
end
