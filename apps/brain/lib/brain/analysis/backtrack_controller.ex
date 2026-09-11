defmodule Brain.Analysis.BacktrackController do
  @moduledoc "Controls backtracking with depth limits and thrash protection.\n\nPrevents oscillation between interpretations by:\n- Limiting maximum backtracks per input (budget)\n- Applying activation cost per backtrack\n- Detecting oscillation patterns\n- Forcing clarification when budget exhausted\n\nThis addresses the \"backtracking loops (thrash risk)\" problem.\n"

  alias Brain.Analysis.{Interpretation, ChunkProfile, SlotDetector, Pipeline, TypeHierarchy}
  require Logger

  @max_backtracks 2
  @backtrack_cost 0.15
  # How many recent turns to consider when checking cross-turn entity-type
  # contradictions (mirrors ContextResolver's default history depth).
  @history_depth 5
  # An entity must be at least this confident to trigger a mismatch backtrack —
  # keeps low-confidence / speculative extractions from forcing a backtrack.
  @entity_conf_floor 0.6

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

  @doc "Checks if an interpretation has contradictions that warrant backtracking.\n\n`opts` may carry `:conversation_history` (a list of recent turn snapshots, each\na map with `:entities`) to enable the cross-turn entity-type check.\n\nReturns {:needs_backtrack, reason} or :ok\n"
  def check_for_contradictions(interp, opts \\ [])

  def check_for_contradictions(%Interpretation{} = interp, opts) do
    checks = [
      &check_missing_required_slots/1,
      fn i -> check_entity_mismatch(i, opts) end,
      &check_confidence_drop/1
    ]

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

  # Two real, guarded signals (was: always :ok, so this check never fired):
  #   1. self-consistency — the interpretation's own entities are ALL
  #      incompatible with the intent domain's expected types (strong sign the
  #      intent is wrong), reusing Pipeline.expected_entity_types_from_domain/1
  #      + TypeHierarchy.compatible?/2 (the pattern from entity_disambiguator).
  #   2. cross-turn — an entity's type contradicts how the SAME value was typed
  #      in a recent turn (reusing the conversation_history snapshot format).
  # Both skip unknown/empty/low-confidence entities to avoid false-positive
  # backtracks; a false positive is bounded anyway (it only promotes a secondary
  # interpretation, and no-ops when there is no alternative).
  defp check_entity_mismatch(%Interpretation{} = interp, opts) do
    with :ok <- self_consistency_mismatch(interp) do
      cross_turn_mismatch(interp, opts)
    end
  end

  defp self_consistency_mismatch(%Interpretation{intent: intent, entities: entities})
       when is_binary(intent) and is_list(entities) and entities != [] do
    with domain when not is_nil(domain) <- domain_from_intent(intent),
         expected when expected != [] <- Pipeline.expected_entity_types_from_domain(domain),
         qualifying when qualifying != [] <- Enum.filter(entities, &qualifies?/1) do
      any_compatible? =
        Enum.any?(qualifying, fn e ->
          Enum.any?(expected, &TypeHierarchy.compatible?(entity_type_of(e), &1))
        end)

      if any_compatible? do
        :ok
      else
        {:contradiction,
         {:entity_mismatch, {:intent_domain, intent, Enum.map(qualifying, &entity_type_of/1)}}}
      end
    else
      _ -> :ok
    end
  end

  defp self_consistency_mismatch(_), do: :ok

  defp cross_turn_mismatch(%Interpretation{entities: entities}, opts) when is_list(entities) do
    prior = recent_entities(Keyword.get(opts, :conversation_history, []))

    if prior == [] do
      :ok
    else
      case Enum.find(entities, fn e -> qualifies?(e) and conflicts_with_history?(e, prior) end) do
        nil ->
          :ok

        e ->
          {:contradiction,
           {:entity_mismatch, {:cross_turn, entity_value_of(e), entity_type_of(e)}}}
      end
    end
  end

  defp cross_turn_mismatch(_, _), do: :ok

  defp conflicts_with_history?(entity, prior) do
    value = entity_value_of(entity) |> String.downcase()
    type = entity_type_of(entity)

    value != "" and
      Enum.any?(prior, fn pe ->
        String.downcase(entity_value_of(pe)) == value and
          entity_type_of(pe) not in ["unknown", ""] and
          not TypeHierarchy.compatible?(type, entity_type_of(pe))
      end)
  end

  defp recent_entities(history) when is_list(history) do
    history
    |> Enum.take(@history_depth)
    |> Enum.flat_map(fn turn -> Map.get(turn, :entities) || [] end)
    |> Enum.filter(&is_map/1)
  end

  defp recent_entities(_), do: []

  # An entity qualifies as evidence only if it has a real type and is confident
  # enough — mirrors the leniency the pipeline applies to OOV/PROPN spans.
  defp qualifies?(entity) do
    entity_type_of(entity) not in ["unknown", ""] and
      entity_conf_of(entity) >= @entity_conf_floor and
      entity_source_of(entity) not in [:pos_tagger_propn, "pos_tagger_propn"]
  end

  defp entity_type_of(e),
    do: to_string(Map.get(e, :entity_type) || "unknown")

  defp entity_value_of(e),
    do: to_string(Map.get(e, :value) || Map.get(e, :text) || "")

  defp entity_conf_of(e), do: Map.get(e, :confidence) || 1.0

  defp entity_source_of(e), do: Map.get(e, :source)

  defp domain_from_intent(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [d, _] -> safe_domain_atom(d)
      _ -> nil
    end
  end

  defp domain_from_intent(_), do: nil

  # Only the closed set of known domains matters (expected_entity_types_from_domain
  # returns [] for anything else); use existing atoms and never leak new ones.
  defp safe_domain_atom(d) do
    String.to_existing_atom(d)
  rescue
    ArgumentError -> nil
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
        humanize_string(label)

      _ ->
        humanize_string(intent)
    end
  end

  # Guard against a nil / non-binary intent reaching String.replace (a
  # clarification can be built from a promoted alternative whose intent is nil).
  defp humanize_string(s) when is_binary(s) and s != "",
    do: s |> String.replace(".", " ") |> String.replace("_", " ")

  defp humanize_string(_), do: "this"

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
