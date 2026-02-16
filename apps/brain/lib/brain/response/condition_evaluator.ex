defmodule Brain.Response.ConditionEvaluator do
  @moduledoc """
  Evaluates condition expressions for conditional template selection.

  Supports simple expressions that reference context signals:
  - `has_entity:type` - Entity of specified type is present
  - `missing_entity:type` - Entity is not present
  - `slot_filled:name` - Slot has a value
  - `slot_missing:name` - Slot is empty
  - `confidence:high/medium/low` - Confidence threshold
  - `speech_act:sub_type` - Speech act matches

  ## Enrichment Conditions

  For templates that require live data from external services:
  - `enriched:field` - Enrichment data contains the specified field
  - `service_available:name` - Service is configured and healthy
  - `enrichment_failed` - Enrichment was attempted but failed
  - `enrichment_success` - Enrichment succeeded

  Compound expressions with AND / OR / NOT:
  - `has_entity:person AND confidence:high`
  - `slot_missing:address OR slot_missing:date-time`
  - `slot_filled:location AND enriched:temperature`
  - `slot_filled:location AND NOT service_available:weather`

  ## Usage

      context = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        filled_slots: ["person"],
        missing_slots: ["location"],
        confidence: 0.85,
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      ConditionEvaluator.evaluate("has_entity:person", context)
      # => true

      ConditionEvaluator.evaluate("has_entity:person AND confidence:high", context)
      # => true

  ## Enrichment Example

      context = %{
        filled_slots: ["location"],
        enriched_data: %{temperature: "72°F", conditions: "sunny"},
        enrichment_status: :success
      }

      ConditionEvaluator.evaluate("enriched:temperature", context)
      # => true

      ConditionEvaluator.evaluate("slot_filled:location AND enriched:temperature", context)
      # => true
  """

  require Logger

  @confidence_thresholds %{
    high: 0.8,
    medium: 0.5,
    low: 0.3
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Evaluates a condition expression against the given context.

  Returns `true` if the condition matches, `false` otherwise.
  Empty or nil conditions always return `true` (backward compatible).
  """
  def evaluate(nil, _context), do: true
  def evaluate("", _context), do: true

  def evaluate(condition, context) when is_binary(condition) do
    condition
    |> String.trim()
    |> parse()
    |> eval_ast(context)
  rescue
    e ->
      Logger.warning("Failed to evaluate condition '#{condition}': #{Exception.message(e)}")
      false
  end

  @doc """
  Parses a condition string into an AST.

  Returns a tuple representing the parsed condition:
  - `{:and, left, right}` - AND expression
  - `{:or, left, right}` - OR expression
  - `{:not, inner}` - NOT expression
  - `{:condition, type, value}` - Simple condition
  """
  def parse(condition) when is_binary(condition) do
    condition = String.trim(condition)

    cond do
      condition == "" ->
        {:always, true}

      String.contains?(condition, " OR ") ->
        parse_or(condition)

      String.contains?(condition, " AND ") ->
        parse_and(condition)

      String.starts_with?(condition, "NOT ") ->
        parse_not(condition)

      true ->
        parse_simple(condition)
    end
  end

  # ============================================================================
  # Parsing
  # ============================================================================

  defp parse_or(condition) do
    # Split on first OR (left-associative)
    case String.split(condition, " OR ", parts: 2) do
      [left, right] ->
        {:or, parse(String.trim(left)), parse(String.trim(right))}

      _ ->
        parse_simple(condition)
    end
  end

  defp parse_and(condition) do
    # Split on first AND (left-associative)
    case String.split(condition, " AND ", parts: 2) do
      [left, right] ->
        {:and, parse(String.trim(left)), parse(String.trim(right))}

      _ ->
        parse_simple(condition)
    end
  end

  defp parse_not(condition) do
    # Remove "NOT " prefix and parse the rest
    inner = String.replace_prefix(condition, "NOT ", "")
    {:not, parse(String.trim(inner))}
  end

  defp parse_simple(condition) do
    case String.split(condition, ":", parts: 2) do
      [type, value] ->
        {:condition, String.trim(type), String.trim(value)}

      [type] ->
        {:condition, String.trim(type), nil}
    end
  end

  # ============================================================================
  # Evaluation
  # ============================================================================

  defp eval_ast({:always, value}, _context), do: value

  defp eval_ast({:and, left, right}, context) do
    eval_ast(left, context) and eval_ast(right, context)
  end

  defp eval_ast({:or, left, right}, context) do
    eval_ast(left, context) or eval_ast(right, context)
  end

  defp eval_ast({:not, inner}, context) do
    not eval_ast(inner, context)
  end

  defp eval_ast({:condition, type, value}, context) do
    eval_condition(type, value, context)
  end

  # ============================================================================
  # Condition Type Handlers
  # ============================================================================

  defp eval_condition("has_entity", entity_type, context) do
    entities = Map.get(context, :entities, [])

    Enum.any?(entities, fn entity ->
      get_entity_type(entity) == entity_type
    end)
  end

  defp eval_condition("missing_entity", entity_type, context) do
    not eval_condition("has_entity", entity_type, context)
  end

  defp eval_condition("slot_filled", slot_name, context) do
    filled_slots = Map.get(context, :filled_slots, [])
    slot_name in filled_slots
  end

  defp eval_condition("slot_missing", slot_name, context) do
    missing_slots = Map.get(context, :missing_slots, [])
    slot_name in missing_slots
  end

  defp eval_condition("confidence", level, context) do
    confidence = Map.get(context, :confidence, 0.0)
    threshold = Map.get(@confidence_thresholds, safe_atom(level), 0.5)

    case level do
      "high" -> confidence >= threshold
      "medium" -> confidence >= threshold and confidence < @confidence_thresholds.high
      "low" -> confidence < @confidence_thresholds.medium
      _ -> confidence >= threshold
    end
  end

  defp eval_condition("speech_act", sub_type, context) do
    speech_act = Map.get(context, :speech_act, %{})
    actual_sub_type = Map.get(speech_act, :sub_type)

    cond do
      is_atom(actual_sub_type) ->
        Atom.to_string(actual_sub_type) == sub_type

      is_binary(actual_sub_type) ->
        actual_sub_type == sub_type

      true ->
        false
    end
  end

  # ============================================================================
  # Enrichment Conditions
  # ============================================================================

  defp eval_condition("enriched", field_name, context) do
    # Check if enrichment data contains the specified field
    enriched_data = Map.get(context, :enriched_data, %{})

    # Support both atom and string keys
    field_key = safe_atom(field_name)

    Map.has_key?(enriched_data, field_key) or Map.has_key?(enriched_data, field_name)
  end

  defp eval_condition("service_available", service_name, context) do
    # Check if a service is configured and available
    # First check context for cached availability
    available_services = Map.get(context, :available_services, [])

    if available_services != [] do
      service_name in available_services or safe_atom(service_name) in available_services
    else
      # Fall back to checking dispatcher
      alias Brain.Services.Dispatcher
      world = Map.get(context, :world_id) || Map.get(context, :world, "default")
      Dispatcher.service_available?(safe_atom(service_name), world: world)
    end
  end

  defp eval_condition("enrichment_failed", _value, context) do
    # Check if enrichment was attempted but failed
    enrichment_status = Map.get(context, :enrichment_status)
    enrichment_status == :failed
  end

  defp eval_condition("enrichment_success", _value, context) do
    # Check if enrichment succeeded
    enrichment_status = Map.get(context, :enrichment_status)
    enrichment_status == :success
  end

  defp eval_condition(unknown_type, _value, _context) do
    Logger.warning("Unknown condition type: #{unknown_type}")
    false
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp safe_atom(val) when is_atom(val), do: val
  defp safe_atom(val) when is_binary(val) do
    String.to_existing_atom(val)
  rescue
    ArgumentError -> :unknown
  end

  defp get_entity_type(entity) when is_map(entity) do
    entity[:entity_type] || entity["entity_type"] || ""
  end

  defp get_entity_type(_), do: ""
end
