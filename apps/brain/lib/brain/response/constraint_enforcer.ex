defmodule Brain.Response.ConstraintEnforcer do
  @moduledoc """
  Structural gate for Ouro-generated responses.

  Only rejects output that is structurally unusable -- empty, too
  long/short, or degenerate repetition. All semantic validation
  (speech act alignment, belief grounding, entity coverage) is
  handled by `ResponseEvaluator` scoring dimensions that the
  `RefinementLoop` can iterate on.
  """

  require Logger

  @max_response_length 2000
  @min_response_length 2

  @doc """
  Validates a generated response against structural constraints only.

  Returns `{:ok, text}` if the response passes all checks,
  or `{:rejected, reason}` if it fails validation.

  Options:
    - `:unified_context` - rich context map (unused by structural checks,
      kept for API compatibility)
  """
  def validate(response_text, _primitives, _opts \\ []) when is_binary(response_text) do
    checks = [
      &check_length/1,
      &check_not_empty/1,
      &check_degenerate_output/1
    ]

    case run_checks(checks, response_text) do
      :ok -> {:ok, response_text}
      {:rejected, reason} -> {:rejected, reason}
    end
  end

  defp run_checks([], _text), do: :ok

  defp run_checks([check | rest], text) do
    case check.(text) do
      :ok -> run_checks(rest, text)
      {:rejected, _} = rejection -> rejection
    end
  end

  defp check_length(text) do
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

  defp check_not_empty(text) do
    trimmed = String.trim(text)

    if trimmed == "" do
      {:rejected, "Empty response"}
    else
      :ok
    end
  end

  defp check_degenerate_output(text) do
    words = String.split(text)
    word_count = length(words)

    if word_count < 6 do
      :ok
    else
      freqs = Enum.frequencies(words)
      most_common_count = freqs |> Map.values() |> Enum.max(fn -> 0 end)
      repetition_ratio = most_common_count / word_count

      if repetition_ratio > 0.4 do
        {:rejected, "Degenerate repetitive output detected (#{Float.round(repetition_ratio * 100, 1)}% repetition)"}
      else
        :ok
      end
    end
  end
end
