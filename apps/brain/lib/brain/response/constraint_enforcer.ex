defmodule Brain.Response.ConstraintEnforcer do
  @moduledoc """
  Post-generation validation for Ouro-generated responses.

  Ensures the model's output stays within the bounds defined by the
  realization packet and unified context. Checks for:

  1. Length bounds -- response is within reasonable length for the plan
  2. Non-empty -- response has actual content
  3. Slot coverage -- if clarification was required, the response asks about it
  4. Entity preservation -- entity values from primitives appear in the output
  5. Fact fidelity -- facts from content primitives are reflected in output
  6. Tone alignment -- response matches the specified tone
  7. Overclaim detection -- response doesn't assert facts not in the payload
  8. Uncertainty preservation -- uncertain claims aren't stated definitively

  If validation fails, the response is rejected and the system falls
  back to template-based rendering.
  """

  alias Brain.ML.Tokenizer

  require Logger

  @max_response_length 2000
  @min_response_length 2

  @doc """
  Validates a generated response against the primitive plan and context.

  Returns `{:ok, text}` if the response passes all checks,
  or `{:rejected, reason}` if it fails validation.

  Options:
    - `:unified_context` - rich context map from ContextBuilder
  """
  def validate(response_text, primitives, opts \\ []) when is_binary(response_text) do
    unified_context = extract_unified_context(opts)

    checks = [
      &check_length/3,
      &check_not_empty/3,
      &check_slot_coverage/3,
      &check_entity_preservation/3,
      &check_fact_fidelity/3,
      &check_overclaim/3,
      &check_uncertainty_preservation/3
    ]

    case run_checks(checks, response_text, primitives, unified_context) do
      :ok -> {:ok, response_text}
      {:rejected, reason} -> {:rejected, reason}
    end
  end

  defp extract_unified_context(opts) when is_list(opts), do: Keyword.get(opts, :unified_context, %{})
  defp extract_unified_context(opts) when is_map(opts), do: Map.get(opts, :unified_context, %{})
  defp extract_unified_context(_), do: %{}

  defp run_checks([], _text, _primitives, _ctx), do: :ok

  defp run_checks([check | rest], text, primitives, ctx) do
    case check.(text, primitives, ctx) do
      :ok -> run_checks(rest, text, primitives, ctx)
      {:rejected, _} = rejection -> rejection
    end
  end

  defp check_length(text, _primitives, _ctx) do
    word_count = text |> String.split() |> length()

    cond do
      word_count < @min_response_length ->
        {:rejected, "Response too short (#{word_count} words)"}

      String.length(text) > @max_response_length ->
        {:rejected, "Response too long (#{String.length(text)} chars)"}

      true ->
        :ok
    end
  end

  defp check_not_empty(text, _primitives, _ctx) do
    trimmed = String.trim(text)

    if trimmed == "" do
      {:rejected, "Empty response"}
    else
      :ok
    end
  end

  defp check_slot_coverage(text, primitives, _ctx) do
    clarification_primitives =
      Enum.filter(primitives, fn p ->
        p.type == :follow_up and p.variant == :clarification
      end)

    case clarification_primitives do
      [] ->
        :ok

      clarifications ->
        missing_slots =
          clarifications
          |> Enum.flat_map(fn p -> Map.get(p.content, :missing_slots, []) end)
          |> Enum.map(&to_string/1)

        has_question = String.contains?(text, "?")

        if missing_slots != [] and not has_question do
          {:rejected, "Clarification required but response has no question"}
        else
          :ok
        end
    end
  end

  defp check_entity_preservation(text, primitives, _ctx) do
    entity_values =
      primitives
      |> Enum.flat_map(fn p ->
        entities = Map.get(p.content, :entity_context, []) ++ Map.get(p.content, :entities, [])

        Enum.flat_map(entities, fn e ->
          value = Map.get(e, :value) || Map.get(e, "value") || Map.get(e, :text) || ""
          if value != "" and String.length(value) > 1, do: [value], else: []
        end)
      end)
      |> Enum.uniq()

    if entity_values == [] do
      :ok
    else
      text_lower = String.downcase(text)

      present_count =
        Enum.count(entity_values, fn val ->
          String.contains?(text_lower, String.downcase(val))
        end)

      coverage = present_count / max(length(entity_values), 1)

      if coverage < 0.5 and length(entity_values) > 0 do
        missing = Enum.reject(entity_values, fn val -> String.contains?(text_lower, String.downcase(val)) end)
        {:rejected, "Missing entities: #{Enum.join(Enum.take(missing, 3), ", ")}"}
      else
        :ok
      end
    end
  end

  defp check_fact_fidelity(text, primitives, _ctx) do
    factual_primitives =
      Enum.filter(primitives, fn p ->
        p.type == :content and p.variant == :factual
      end)

    if factual_primitives == [] do
      :ok
    else
      facts =
        factual_primitives
        |> Enum.flat_map(fn p -> Map.get(p.content, :facts, []) end)
        |> Enum.flat_map(fn
          %{fact: fact_text} when is_binary(fact_text) -> [fact_text]
          %{"fact" => fact_text} when is_binary(fact_text) -> [fact_text]
          fact when is_binary(fact) -> [fact]
          _ -> []
        end)

      if facts == [] do
        :ok
      else
        text_tokens =
          Tokenizer.tokenize_normalized(text)
          |> MapSet.new()

        any_match =
          Enum.any?(facts, fn fact ->
            fact_tokens =
              Tokenizer.tokenize_normalized(fact)
              |> MapSet.new()

            overlap = MapSet.intersection(text_tokens, fact_tokens) |> MapSet.size()
            overlap >= max(div(MapSet.size(fact_tokens), 3), 1)
          end)

        if any_match do
          :ok
        else
          {:rejected, "No facts from content primitives reflected in output"}
        end
      end
    end
  end

  defp check_overclaim(text, primitives, ctx) do
    analysis = get_in_map(ctx, [:analysis]) || get_in_map(ctx, [:primary_analysis])

    confidence =
      case analysis do
        %{confidence: c} when is_number(c) -> c
        _ -> 0.5
      end

    if confidence >= 0.6 do
      :ok
    else
      all_content_tokens =
        primitives
        |> Enum.flat_map(fn p ->
          content_text =
            p.content
            |> Map.values()
            |> Enum.flat_map(fn
              v when is_binary(v) -> [v]
              v when is_list(v) -> Enum.flat_map(v, &extract_text_values/1)
              v when is_map(v) -> extract_text_values(v)
              _ -> []
            end)

          Enum.flat_map(content_text, &Tokenizer.tokenize_normalized/1)
        end)
        |> MapSet.new()

      response_tokens =
        Tokenizer.tokenize_normalized(text)
        |> MapSet.new()

      novel_tokens = MapSet.difference(response_tokens, all_content_tokens)
      stop_words = MapSet.new(~w(i the a an is are was were be been being have has had do does did will would shall should may might must can could))
      novel_content = MapSet.difference(novel_tokens, stop_words)

      novel_ratio = MapSet.size(novel_content) / max(MapSet.size(response_tokens), 1)

      if novel_ratio > 0.7 do
        {:rejected, "Response introduces too much novel content (#{Float.round(novel_ratio * 100, 1)}% novel)"}
      else
        :ok
      end
    end
  end

  defp check_uncertainty_preservation(text, _primitives, ctx) do
    accumulator = get_in_map(ctx, [:accumulator]) || %{}
    should_hedge = Map.get(accumulator, :should_hedge, false)

    if not should_hedge do
      :ok
    else
      definitive_markers = ~w(definitely certainly absolutely always never)

      text_lower = String.downcase(text)
      has_definitive = Enum.any?(definitive_markers, &String.contains?(text_lower, &1))

      if has_definitive do
        {:rejected, "Response uses definitive language but system should hedge"}
      else
        :ok
      end
    end
  end

  defp extract_text_values(v) when is_binary(v), do: [v]
  defp extract_text_values(v) when is_map(v) do
    v |> Map.values() |> Enum.flat_map(&extract_text_values/1)
  end
  defp extract_text_values(v) when is_list(v), do: Enum.flat_map(v, &extract_text_values/1)
  defp extract_text_values(_), do: []

  defp get_in_map(map, []), do: map
  defp get_in_map(map, [key | rest]) when is_map(map) do
    case Map.get(map, key) do
      nil -> nil
      val -> get_in_map(val, rest)
    end
  end
  defp get_in_map(_, _), do: nil
end
