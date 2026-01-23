defmodule ChatBot.Analysis.BacktrackController do
  @moduledoc """
  Controls backtracking with depth limits and thrash protection.

  Prevents oscillation between interpretations by:
  - Limiting maximum backtracks per input (budget)
  - Applying activation cost per backtrack
  - Detecting oscillation patterns
  - Forcing clarification when budget exhausted

  This addresses the "backtracking loops (thrash risk)" problem.
  """

  alias ChatBot.Analysis.Interpretation

  require Logger

  @max_backtracks 2
  @backtrack_cost 0.15

  defstruct [
    :input_text,
    backtrack_count: 0,
    demoted_interpretations: [],
    total_cost_incurred: 0.0,
    oscillation_detected: false,
    interpretation_history: []
  ]

  @type t :: %__MODULE__{
          input_text: String.t(),
          backtrack_count: non_neg_integer(),
          demoted_interpretations: list(map()),
          total_cost_incurred: float(),
          oscillation_detected: boolean(),
          interpretation_history: list(String.t())
        }

  @doc """
  Creates a new backtrack controller for an input.
  """
  def new(input_text) do
    %__MODULE__{input_text: input_text}
  end

  @doc """
  Checks if an interpretation has contradictions that warrant backtracking.

  Returns {:needs_backtrack, reason} or :ok
  """
  def check_for_contradictions(%Interpretation{} = interp) do
    checks = [
      &check_missing_required_slots/1,
      &check_entity_mismatch/1,
      &check_confidence_drop/1
    ]

    Enum.find_value(checks, :ok, fn check ->
      case check.(interp) do
        {:contradiction, reason} -> {:needs_backtrack, reason}
        :ok -> nil
      end
    end)
  end

  @doc """
  Attempts to backtrack, promoting a secondary interpretation.

  Returns:
  - {:ok, new_state, new_interpretation, cost} on success
  - {:force_clarification, clarification_prompt} if budget exhausted
  - {:error, :no_alternatives} if no alternatives available
  """
  def attempt_backtrack(%__MODULE__{} = state, %Interpretation{} = interp, reason) do
    cond do
      # Budget exhausted
      state.backtrack_count >= @max_backtracks ->
        clarification = build_clarification_from_ambiguity(state, interp)

        Logger.info("Backtrack budget exhausted, forcing clarification", %{
          backtracks: state.backtrack_count,
          reason: reason
        })

        {:force_clarification, clarification}

      # Oscillation detected
      detect_oscillation?(state, interp) ->
        Logger.warning("Oscillation detected, forcing clarification", %{
          history: state.interpretation_history
        })

        clarification = build_oscillation_clarification(state, interp)
        {:force_clarification, clarification}

      # No alternatives to promote
      interp.alternatives == [] ->
        {:error, :no_alternatives}

      # Proceed with backtrack
      true ->
        perform_backtrack(state, interp, reason)
    end
  end

  @doc """
  Returns backtracking statistics for self-reflection.
  """
  def stats(%__MODULE__{} = state) do
    %{
      backtrack_count: state.backtrack_count,
      budget_remaining: @max_backtracks - state.backtrack_count,
      total_cost: state.total_cost_incurred,
      oscillation_detected: state.oscillation_detected,
      demoted_count: length(state.demoted_interpretations)
    }
  end

  @doc """
  Checks if we should ask for clarification instead of backtracking again.

  Useful for stability self-reflection (ST6, B8).
  """
  def should_clarify?(%__MODULE__{} = state) do
    state.backtrack_count >= @max_backtracks - 1 or state.oscillation_detected
  end

  @doc """
  Returns the maximum allowed backtracks.
  """
  def max_backtracks, do: @max_backtracks

  @doc """
  Returns the cost per backtrack.
  """
  def backtrack_cost, do: @backtrack_cost

  # Private functions - Contradiction checks

  defp check_missing_required_slots(%Interpretation{slots: nil}), do: :ok

  defp check_missing_required_slots(%Interpretation{slots: slots, intent: intent}) do
    if slots.all_required_filled do
      :ok
    else
      missing = slots.missing_required

      Logger.debug("Contradiction: missing required slots", %{
        intent: intent,
        missing: missing
      })

      {:contradiction, {:missing_required, missing}}
    end
  end

  defp check_entity_mismatch(%Interpretation{} = interp) do
    # Check if entities don't make sense for the intent
    entities = interp.entities || []
    intent = interp.intent || ""

    mismatches =
      cond do
        # Weather intent but no location, time entities
        String.contains?(intent, "weather") and
            not has_entity_type?(entities, ["location", "city", "place-name"]) ->
          # This is borderline - might be clarification-worthy but not a hard contradiction
          nil

        # Device control but no device entity
        String.contains?(intent, "device") and
            not has_entity_type?(entities, ["device", "light", "switch"]) ->
          {:entity_mismatch, "device intent without device entity"}

        # Music intent with location entity (suspicious)
        String.contains?(intent, "music") and
            has_entity_type?(entities, ["location"]) and
            not has_entity_type?(entities, ["music-artist", "song", "album"]) ->
          {:entity_mismatch, "music intent with location but no music entities"}

        true ->
          nil
      end

    case mismatches do
      nil -> :ok
      reason -> {:contradiction, reason}
    end
  end

  defp check_confidence_drop(%Interpretation{activation: activation}) do
    # Very low activation suggests something is wrong
    if activation < 0.2 do
      {:contradiction, {:low_confidence, activation}}
    else
      :ok
    end
  end

  defp has_entity_type?(entities, types) when is_list(types) do
    Enum.any?(entities, fn e ->
      entity_type = e[:entity] || e["entity"] || e[:type] || e["type"]
      entity_type in types
    end)
  end

  # Private functions - Backtracking

  defp perform_backtrack(state, interp, reason) do
    # Promote the alternative
    case Interpretation.promote_alternative(interp) do
      {:ok, promoted} ->
        # Record the demotion
        demoted = %{
          intent: interp.intent,
          activation: interp.activation,
          reason: reason
        }

        # Apply cost to promoted interpretation
        penalized_activation = max(0.1, promoted.activation - @backtrack_cost)

        promoted_with_cost = %{promoted | activation: penalized_activation}

        # Update state
        new_state = %{
          state
          | backtrack_count: state.backtrack_count + 1,
            total_cost_incurred: state.total_cost_incurred + @backtrack_cost,
            demoted_interpretations: [demoted | state.demoted_interpretations],
            interpretation_history: [interp.intent | state.interpretation_history]
        }

        Logger.debug("Backtracked interpretation", %{
          from: interp.intent,
          to: promoted_with_cost.intent,
          reason: reason,
          new_activation: penalized_activation
        })

        {:ok, new_state, promoted_with_cost, @backtrack_cost}

      {:error, :no_alternatives} ->
        {:error, :no_alternatives}
    end
  end

  defp detect_oscillation?(state, interp) do
    # Check if we're bouncing between the same interpretations
    # We need to check if the alternative we're about to promote
    # was already visited in the history

    next_intent =
      case interp.alternatives do
        [%{intent: intent} | _] -> intent
        _ -> nil
      end

    if next_intent && length(state.interpretation_history) >= 1 do
      # Check if we're about to promote to something we already tried
      # Also include current intent in check (A -> B -> A pattern)
      full_history = [interp.intent | state.interpretation_history]
      next_intent in full_history
    else
      false
    end
  end

  # Private functions - Clarification building

  defp build_clarification_from_ambiguity(state, interp) do
    # Get the competing interpretations
    candidates =
      [interp.intent | Enum.map(state.demoted_interpretations, & &1.intent)]
      |> Enum.uniq()
      |> Enum.take(3)

    case candidates do
      [a, b] ->
        %{
          type: :disambiguation,
          prompt: "I'm not sure if you're asking about #{humanize_intent(a)} or #{humanize_intent(b)}. Could you clarify?",
          options: [a, b]
        }

      [a, b, c] ->
        %{
          type: :disambiguation,
          prompt:
            "I'm having trouble understanding. Are you asking about #{humanize_intent(a)}, #{humanize_intent(b)}, or #{humanize_intent(c)}?",
          options: [a, b, c]
        }

      [_single] ->
        # Only one candidate - ask about missing info
        build_missing_info_clarification(interp)

      [] ->
        %{
          type: :general,
          prompt: "I'm having trouble understanding. Could you rephrase that?",
          options: []
        }
    end
  end

  defp build_oscillation_clarification(state, interp) do
    oscillating =
      [interp.intent | state.interpretation_history]
      |> Enum.uniq()
      |> Enum.take(2)

    case oscillating do
      [a, b] ->
        %{
          type: :oscillation,
          prompt:
            "I keep going back and forth between understanding this as #{humanize_intent(a)} and #{humanize_intent(b)}. Which did you mean?",
          options: [a, b]
        }

      _ ->
        %{
          type: :general,
          prompt: "I'm having trouble pinning down what you mean. Could you rephrase?",
          options: []
        }
    end
  end

  defp build_missing_info_clarification(interp) do
    missing = Interpretation.missing_required(interp)

    case missing do
      [slot | _] ->
        %{
          type: :missing_slot,
          prompt: generate_slot_prompt(slot, interp.intent),
          missing_slot: slot
        }

      [] ->
        %{
          type: :general,
          prompt: "Could you provide more details?",
          options: []
        }
    end
  end

  defp generate_slot_prompt(slot, intent) do
    case {slot, intent} do
      {"location", "weather.query"} ->
        "What location would you like the weather for?"

      {"location", _} ->
        "Which location are you referring to?"

      {"device", "device.control"} ->
        "Which device would you like me to control?"

      {"action", "device.control"} ->
        "What would you like me to do with the device?"

      {"date", _} ->
        "For which date?"

      {"time", _} ->
        "At what time?"

      {slot_name, _} ->
        readable = slot_name |> String.replace("-", " ") |> String.replace("_", " ")
        "Could you please specify the #{readable}?"
    end
  end

  defp humanize_intent(nil), do: "something"
  defp humanize_intent(""), do: "something"

  defp humanize_intent(intent) when is_binary(intent) do
    intent
    |> String.replace(".", " ")
    |> String.replace("_", " ")
    |> String.replace("smalltalk ", "")
  end
end
