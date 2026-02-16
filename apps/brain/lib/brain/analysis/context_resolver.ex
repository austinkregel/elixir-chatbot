defmodule Brain.Analysis.ContextResolver do
  @moduledoc """
  Attempts to fill missing slots from available context sources.

  Context sources (in priority order):
  1. Current message - explicitly stated (already handled by SlotDetector)
  2. Conversation history - recently mentioned (within N turns)
  3. User model - learned facts from epistemic system (with confidence thresholds)
  4. User profile - stored preferences (location, timezone, etc.)
  5. Defaults - schema-defined defaults (already handled by SlotDetector)

  This module focuses on sources 2, 3, and 4.
  """

  alias Brain.Analysis.SlotResult
  alias Brain.Epistemic.UserModelStore
  alias Brain.Epistemic.Types.Config
  alias Brain.ML.Gazetteer

  require Logger

  # How many conversation turns to look back for context
  @default_history_depth 5

  # Minimum confidence threshold for user model facts
  @user_model_confidence_threshold 0.6

  # Slots that commonly come from user profile
  @profile_slots ~w(location timezone preferred_temperature_unit preferred_language)

  # Fallback mapping from slot names to user model predicates
  # Used when Gazetteer lookup fails
  @slot_to_predicate_fallback %{
    "location" => [:location, :city, :home_city, :preferred_location],
    "timezone" => [:timezone],
    "preferred_language" => [:language, :preferred_language],
    "preferred_temperature_unit" => [:temperature_unit],
    "name" => [:name],
    "workplace" => [:workplace, :employer, :company]
  }

  @doc """
  Resolves missing slots from conversation history, user model, and user profile.

  Options:
  - :conversation_history - list of previous messages with entities
  - :user_profile - map of user preferences
  - :user_id - user ID for querying the epistemic user model
  - :history_depth - how many turns to look back (default: 5)

  Returns an updated SlotResult with resolved slots.
  """
  def resolve(%SlotResult{} = slot_result, opts \\ []) do
    history = Keyword.get(opts, :conversation_history, [])
    profile = Keyword.get(opts, :user_profile, %{})
    user_id = Keyword.get(opts, :user_id)
    depth = Keyword.get(opts, :history_depth, @default_history_depth)

    # Get missing slots that need resolution
    missing = slot_result.missing_required ++ slot_result.missing_optional

    if missing == [] do
      slot_result
    else
      slot_result
      |> resolve_from_history(missing, history, depth)
      |> resolve_from_user_model(user_id)
      |> resolve_from_profile(profile)
    end
  end

  @doc """
  Extracts potential context from a message for future reference.

  This creates a context snapshot that can be stored in conversation history.
  """
  def extract_context(entities, intent, timestamp \\ nil) do
    timestamp = timestamp || System.system_time(:millisecond)

    # Group entities by type
    entity_map =
      entities
      |> Enum.group_by(fn e -> e[:entity_type] end)
      |> Enum.map(fn {type, ents} ->
        # Take the highest confidence one
        best = Enum.max_by(ents, fn e -> e[:confidence] || e["confidence"] || 0 end)
        {type, best[:value] || best["value"]}
      end)
      |> Enum.into(%{})

    %{
      entities: entity_map,
      intent: intent,
      timestamp: timestamp
    }
  end

  @doc """
  Generates clarification prompts for missing required slots.
  """
  def generate_clarification_prompts(%SlotResult{} = slot_result, schema \\ nil) do
    templates =
      if schema do
        Map.get(schema, "clarification_templates", %{})
      else
        %{}
      end

    slot_result.missing_required
    |> Enum.map(fn slot_name ->
      case Map.get(templates, slot_name) do
        nil -> generate_default_prompt(slot_name)
        template -> template
      end
    end)
  end

  # Private functions

  defp resolve_from_history(slot_result, missing, history, depth) do
    # Take only recent history
    recent_history = Enum.take(history, depth)

    # Try to find each missing slot in history
    Enum.reduce(missing, slot_result, fn slot_name, acc ->
      if Map.has_key?(acc.filled_slots, slot_name) do
        # Already filled
        acc
      else
        case find_in_history(slot_name, recent_history) do
          {:ok, value, turns_ago} ->
            # Confidence decreases with age
            confidence = calculate_history_confidence(turns_ago, depth)
            SlotResult.fill_slot(acc, slot_name, value, :conversation, confidence)

          :not_found ->
            acc
        end
      end
    end)
  end

  defp find_in_history(slot_name, history) do
    # Look through history for matching entity
    history
    |> Enum.with_index(1)
    |> Enum.find_value(fn {context, turns_ago} ->
      entities = Map.get(context, :entities, %{})

      # Check for exact slot name match
      case Map.get(entities, slot_name) do
        nil ->
          # Try related entity types
          case find_related_entity(slot_name, entities) do
            nil -> nil
            value -> {:ok, value, turns_ago}
          end

        value ->
          {:ok, value, turns_ago}
      end
    end) || :not_found
  end

  @external_resource Path.join(:code.priv_dir(:brain), "analysis/related_slot_mappings.json")
  @related_slot_mappings Path.join(:code.priv_dir(:brain), "analysis/related_slot_mappings.json")
                         |> File.read!()
                         |> Jason.decode!()

  defp find_related_entity(slot_name, entities) do
    related_types = Map.get(@related_slot_mappings, slot_name, [])

    Enum.find_value(related_types, fn type ->
      Map.get(entities, type)
    end)
  end

  defp calculate_history_confidence(turns_ago, max_depth) do
    # Linear decay from 0.9 to 0.5 based on age
    base = 0.9
    min_conf = 0.5
    decay_rate = (base - min_conf) / max_depth

    max(min_conf, base - (turns_ago - 1) * decay_rate)
  end

  # Resolve slots from the epistemic user model
  defp resolve_from_user_model(slot_result, nil), do: slot_result

  defp resolve_from_user_model(slot_result, user_id) do
    # Skip if epistemic system is disabled
    unless Config.enabled?() do
      slot_result
    else
      missing = slot_result.missing_required ++ slot_result.missing_optional

      Enum.reduce(missing, slot_result, fn slot_name, acc ->
        if Map.has_key?(acc.filled_slots, slot_name) do
          # Already filled
          acc
        else
          case find_in_user_model(slot_name, user_id) do
            {:ok, value, confidence} ->
              Logger.debug("Resolved slot from user model", %{
                slot: slot_name,
                value: value,
                confidence: confidence
              })

              SlotResult.fill_slot(acc, slot_name, value, :user_model, confidence)

            :not_found ->
              acc
          end
        end
      end)
    end
  end

  defp find_in_user_model(slot_name, user_id) do
    # Get predicates to search for this slot via Gazetteer (data-driven)
    predicates = get_user_model_predicates(slot_name)

    # Try each predicate until we find a match
    Enum.find_value(predicates, :not_found, fn predicate ->
      case UserModelStore.get_fact(user_id, predicate) do
        %{value: value, confidence: conf} when conf >= @user_model_confidence_threshold ->
          {:ok, value, conf}

        _ ->
          nil
      end
    end)
  rescue
    _ -> :not_found
  end

  # Look up user model predicates from Gazetteer (slot_mappings.json)
  defp get_user_model_predicates(slot_name) do
    case Gazetteer.lookup(slot_name) do
      {:ok, %{entity_type: "slot_mapping", metadata: meta}} ->
        # Get predicates from the slot mapping
        keys = meta["user_model_keys"] || meta[:user_model_keys] || []
        Enum.map(keys, &safe_atom_key/1)

      _ ->
        # Fall back to hardcoded mappings
        Map.get(@slot_to_predicate_fallback, slot_name, [safe_atom_key(slot_name)])
    end
  end

  defp resolve_from_profile(slot_result, profile) when map_size(profile) == 0 do
    slot_result
  end

  defp resolve_from_profile(slot_result, profile) do
    # Check profile slots
    Enum.reduce(@profile_slots, slot_result, fn slot_name, acc ->
      if Map.has_key?(acc.filled_slots, slot_name) do
        # Already filled
        acc
      else
        case Map.get(profile, slot_name) || Map.get(profile, safe_atom_key(slot_name)) do
          nil ->
            acc

          value ->
            SlotResult.fill_slot(acc, slot_name, value, :user_profile, 0.85)
        end
      end
    end)

    # Also check for any missing slots that might be in profile
    |> resolve_remaining_from_profile(profile)
  end

  defp resolve_remaining_from_profile(slot_result, profile) do
    missing = slot_result.missing_required ++ slot_result.missing_optional

    Enum.reduce(missing, slot_result, fn slot_name, acc ->
      if Map.has_key?(acc.filled_slots, slot_name) do
        acc
      else
        # Check both string and atom keys
        value =
          Map.get(profile, slot_name) ||
            Map.get(profile, safe_atom_key(slot_name))

        case value do
          nil -> acc
          v -> SlotResult.fill_slot(acc, slot_name, v, :user_profile, 0.85)
        end
      end
    end)
  end

  defp safe_atom_key(val) when is_atom(val), do: val
  defp safe_atom_key(val) when is_binary(val) do
    String.to_existing_atom(val)
  rescue
    ArgumentError -> :unknown
  end

  defp generate_default_prompt(slot_name) do
    # Generate a friendly prompt based on slot name
    readable_name =
      slot_name
      |> String.replace("-", " ")
      |> String.replace("_", " ")

    case slot_name do
      "location" -> "What location are you interested in?"
      "device" -> "Which device would you like me to work with?"
      "action" -> "What action would you like to take?"
      "date" -> "For which date?"
      "time" -> "At what time?"
      "account" -> "Which account are you referring to?"
      "content" -> "What would you like the content to be?"
      "query" -> "What would you like to search for?"
      _ -> "Could you please specify the #{readable_name}?"
    end
  end
end
