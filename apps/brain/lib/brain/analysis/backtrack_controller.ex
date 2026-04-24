defmodule Brain.Analysis.BacktrackController do
  @moduledoc "Controls backtracking with depth limits and thrash protection.\n\nPrevents oscillation between interpretations by:\n- Limiting maximum backtracks per input (budget)\n- Applying activation cost per backtrack\n- Detecting oscillation patterns\n- Forcing clarification when budget exhausted\n\nThis addresses the \"backtracking loops (thrash risk)\" problem.\n"

  alias Brain.Analysis.{Interpretation, ChunkProfile, SlotDetector}
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

  @doc "Creates a new backtrack controller for an input.\n"
  def new(input_text) do
    %__MODULE__{input_text: input_text}
  end

  @doc "Checks if an interpretation has contradictions that warrant backtracking.\n\nReturns {:needs_backtrack, reason} or :ok\n"
  def check_for_contradictions(%Interpretation{} = interp) do
    checks = [&check_missing_required_slots/1, &check_entity_mismatch/1, &check_confidence_drop/1]

    Enum.find_value(checks, :ok, fn check ->
      case check.(interp) do
        {:contradiction, reason} -> {:needs_backtrack, reason}
        :ok -> nil
      end
    end)
  end

  @doc "Attempts to backtrack, promoting a secondary interpretation.\n\nReturns:\n- {:ok, new_state, new_interpretation, cost} on success\n- {:force_clarification, clarification_prompt} if budget exhausted\n- {:error, :no_alternatives} if no alternatives available\n"
  def attempt_backtrack(%__MODULE__{} = state, %Interpretation{} = interp, reason) do
    cond do
      state.backtrack_count >= @max_backtracks ->
        clarification = build_clarification_from_ambiguity(state, interp)

        Logger.info("Backtrack budget exhausted, forcing clarification", %{
          backtracks: state.backtrack_count,
          reason: reason
        })

        {:force_clarification, clarification}

      detect_oscillation?(state, interp) ->
        Logger.warning("Oscillation detected, forcing clarification", %{
          history: state.interpretation_history
        })

        clarification = build_oscillation_clarification(state, interp)
        {:force_clarification, clarification}

      interp.alternatives == [] ->
        {:error, :no_alternatives}

      true ->
        perform_backtrack(state, interp, reason)
    end
  end

  @doc "Returns backtracking statistics for self-reflection.\n"
  def stats(%__MODULE__{} = state) do
    %{
      backtrack_count: state.backtrack_count,
      budget_remaining: @max_backtracks - state.backtrack_count,
      total_cost: state.total_cost_incurred,
      oscillation_detected: state.oscillation_detected,
      demoted_count: length(state.demoted_interpretations)
    }
  end

  @doc "Checks if we should ask for clarification instead of backtracking again.\n\nUseful for stability self-reflection (ST6, B8).\n"
  def should_clarify?(%__MODULE__{} = state) do
    state.backtrack_count >= @max_backtracks - 1 or state.oscillation_detected
  end

  @doc "Returns the maximum allowed backtracks.\n"
  def max_backtracks do
    @max_backtracks
  end

  @doc "Returns the cost per backtrack.\n"
  def backtrack_cost do
    @backtrack_cost
  end

  defp check_missing_required_slots(%Interpretation{slots: nil}) do
    :ok
  end

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

  defp check_entity_mismatch(%Interpretation{} = _interp) do
    :ok
  end

  defp check_confidence_drop(%Interpretation{activation: activation}) do
    if activation < 0.2 do
      {:contradiction, {:low_confidence, activation}}
    else
      :ok
    end
  end

  defp perform_backtrack(state, interp, reason) do
    case Interpretation.promote_alternative(interp) do
      {:ok, promoted} ->
        demoted = %{
          intent: interp.intent,
          activation: interp.activation,
          reason: reason
        }

        penalized_activation = max(0.1, promoted.activation - @backtrack_cost)

        promoted_with_cost = %{promoted | activation: penalized_activation}

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
    next_intent =
      case interp.alternatives do
        [%{intent: intent} | _] -> intent
        _ -> nil
      end

    if next_intent && state.interpretation_history != [] do
      full_history = [interp.intent | state.interpretation_history]
      next_intent in full_history
    else
      false
    end
  end

  defp build_clarification_from_ambiguity(state, interp) do
    profile = Map.get(interp, :profile)
    candidates =
      [interp.intent | Enum.map(state.demoted_interpretations, & &1.intent)]
      |> Enum.uniq()
      |> Enum.take(3)

    case candidates do
      [a, b] ->
        %{
          type: :disambiguation,
          prompt:
            "I'm not sure if you're asking about #{humanize_intent(a, profile)} or #{humanize_intent(b)}. Could you clarify?",
          options: [a, b]
        }

      [a, b, c] ->
        %{
          type: :disambiguation,
          prompt:
            "I'm having trouble understanding. Are you asking about #{humanize_intent(a, profile)}, #{humanize_intent(b)}, or #{humanize_intent(c)}?",
          options: [a, b, c]
        }

      [_single] ->
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
    profile = Map.get(interp, :profile)
    oscillating =
      [interp.intent | state.interpretation_history]
      |> Enum.uniq()
      |> Enum.take(2)

    case oscillating do
      [a, b] ->
        %{
          type: :oscillation,
          prompt:
            "I keep going back and forth between understanding this as #{humanize_intent(a, profile)} and #{humanize_intent(b)}. Which did you mean?",
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

  defp humanize_intent(intent, profile \\ nil) do
    case profile do
      %ChunkProfile{derived_label: label} when is_binary(label) and label != "" ->
        label
        |> String.replace(".", " ")
        |> String.replace("_", " ")

      _ ->
        intent |> String.replace(".", " ") |> String.replace("_", " ")
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
    SlotDetector.get_clarification_prompt(slot, intent)
  end
end