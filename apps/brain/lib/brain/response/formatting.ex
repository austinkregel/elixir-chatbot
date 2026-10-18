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
  The words a value is said as in a reply, or nil when it is none: a
  non-blank string as it is, a number as written, and a non-empty list of
  those joined as `format_value/1` joins one. `nil`, a blank string, `[]`, a
  boolean, any other atom (a label, such as `capability: :unknown`), a map,
  and a list holding anything else are not words.

  The one rule both the synthesizer and the lattice realizer fill a
  `$placeholder` by (`Brain.Response.Synthesizer`,
  `Brain.Response.PhraseLatticeRealizer`).
  """
  @spec sayable(term()) :: String.t() | nil
  def sayable(value) when is_list(value) do
    if value != [] and Enum.all?(value, &(sayable(&1) != nil and not is_list(&1))), do: format_value(value)
  end

  def sayable(value) when is_binary(value), do: if(String.trim(value) == "", do: nil, else: value)
  def sayable(value) when is_number(value), do: format_value(value)
  def sayable(_value), do: nil

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
