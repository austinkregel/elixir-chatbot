defmodule Brain.Response.Formatting do
  @moduledoc """
  Shared response formatting utilities.

  Provides `format_value/1` for converting various Elixir types to
  human-readable string representations, and `substitute_placeholders/2`
  for enrichment slot filling in response templates.

  Previously duplicated in Synthesizer and Enricher.
  """

  @doc """
  Formats a value for human-readable display in responses.

  Handles binaries, atoms, numbers, lists, and falls back to `inspect/1`.
  """
  @spec format_value(term()) :: String.t()
  def format_value(value) when is_binary(value), do: value
  def format_value(value) when is_atom(value), do: Atom.to_string(value)
  def format_value(value) when is_number(value), do: to_string(value)
  def format_value(value) when is_list(value), do: Enum.join(value, ", ")
  def format_value(value), do: inspect(value)

  @doc """
  Substitutes enrichment placeholders ($key and @key) in text with values
  from the enriched_data map. Skips nested map values.
  """
  @spec substitute_placeholders(String.t(), map()) :: String.t()
  def substitute_placeholders(text, enriched_data) when is_binary(text) do
    Enum.reduce(enriched_data, text, fn {key, value}, acc ->
      if is_map(value) do
        acc
      else
        key_str = to_string(key)
        value_str = format_value(value)

        acc
        |> String.replace("$#{key_str}", value_str)
        |> String.replace("@#{key_str}", value_str)
      end
    end)
  end
end
