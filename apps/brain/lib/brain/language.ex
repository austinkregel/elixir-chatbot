defmodule Brain.Language do
  @moduledoc """
  Which natural language a request is analyzed in.

  Tokenizing, contractions and chunking are per-language code: each language is
  supported only once code has been written for it. English is the one that
  exists today, so `supported/0` is `["en"]`.

  The language comes from the browser. Only its top `Accept-Language` choice
  counts -- later entries are never consulted -- and a regional variant counts
  as its base language (`en-GB` and `en-US` are English). English applies only
  when the browser names no language at all. A language with no implementation
  raises, naming it, rather than being analyzed as English.

  A world's language is metadata about the world's data and does not select or
  limit the language applied here.
  """

  @supported ["en"]
  @default "en"

  @doc "The language codes with an implementation."
  @spec supported() :: [String.t()]
  def supported, do: @supported

  @doc """
  The language to analyze in, from an `Accept-Language` header value.

  The top choice is the entry with the highest quality (`q`, 1 when absent);
  among equal qualities the earliest wins. `nil`, an empty header, or a top
  choice of `*` names no language, which means English. Raises `ArgumentError`
  for a language with no implementation, or for a header that cannot be read.
  """
  @spec resolve!(String.t() | nil) :: String.t()
  def resolve!(nil), do: @default

  def resolve!(header) when is_binary(header) do
    case top_choice(header) do
      nil -> @default
      "*" -> @default
      tag -> tag |> base_language() |> validate!()
    end
  end

  @doc "Returns `code` when it is a supported language code, and raises otherwise."
  @spec validate!(String.t()) :: String.t()
  def validate!(code) when code in @supported, do: code

  def validate!(code) do
    raise ArgumentError,
          "Language #{inspect(code)} is not supported. Supported languages: #{Enum.join(@supported, ", ")}"
  end

  defp top_choice(header) do
    header
    |> String.split(",", trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.with_index()
    |> Enum.map(fn {entry, index} -> {parse_entry(entry, header), index} end)
    |> Enum.sort_by(fn {{_tag, q}, index} -> {-q, index} end)
    |> case do
      [] -> nil
      [{{tag, _q}, _index} | _] -> tag
    end
  end

  defp parse_entry(entry, header) do
    [tag | params] = entry |> String.split(";") |> Enum.map(&String.trim/1)

    q =
      Enum.find_value(params, 1.0, fn param ->
        case String.split(param, "=", parts: 2) do
          ["q", value] -> parse_quality(value, header)
          _ -> nil
        end
      end)

    {String.downcase(tag), q}
  end

  defp parse_quality(value, header) do
    case Float.parse(value) do
      {q, ""} when q >= 0.0 and q <= 1.0 ->
        q

      _ ->
        raise ArgumentError, "Accept-Language #{inspect(header)} has an unreadable quality #{inspect(value)}"
    end
  end

  defp base_language(tag), do: tag |> String.split("-") |> hd()
end
