defmodule Brain.ML.Lexicon.WordNetParser do
  @moduledoc """
  Parses WordNet Prolog database files into Elixir data structures.

  Each Prolog file contains facts in the format:
      operator(field1,field2,...,fieldn).

  Single quotes enclose strings; doubled single quotes represent literal quotes.
  No spaces between fields. Lines are terminated with a period and newline.
  """

  require Logger

  @ss_type_map %{
    "n" => :noun,
    "v" => :verb,
    "a" => :adj,
    "s" => :adj_satellite,
    "r" => :adv
  }

  @doc """
  Parses wn_s.pl (word senses).

  Each line: s(synset_id,w_num,'word',ss_type,sense_number,tag_count).

  Returns a list of sense maps.
  """
  def parse_senses(path) do
    path
    |> stream_lines()
    |> Enum.reduce([], fn line, acc ->
      case parse_sense_line(line) do
        {:ok, sense} -> [sense | acc]
        :skip -> acc
      end
    end)
    |> Enum.reverse()
  end

  defp parse_sense_line(line) do
    case extract_args(line, "s") do
      nil ->
        :skip

      args_str ->
        case parse_sense_args(args_str) do
          {:ok, _} = result -> result
          _ -> :skip
        end
    end
  end

  defp parse_sense_args(args_str) do
    case split_prolog_args(args_str) do
      [synset_id_str, w_num_str, word_str, ss_type, sense_num_str, tag_count_str] ->
        {:ok,
         %{
           synset_id: parse_int(synset_id_str),
           w_num: parse_int(w_num_str),
           word: unquote_prolog(word_str),
           ss_type: Map.get(@ss_type_map, ss_type, :unknown),
           sense_number: parse_int(sense_num_str),
           tag_count: parse_int(tag_count_str)
         }}

      _ ->
        :error
    end
  end

  @doc """
  Parses wn_g.pl (glosses/definitions).

  Each line: g(synset_id,'(definition) optional example').

  Returns a list of `{synset_id, definition}` tuples.
  """
  def parse_glosses(path) do
    path
    |> stream_lines()
    |> Enum.reduce([], fn line, acc ->
      case parse_gloss_line(line) do
        {:ok, synset_id, gloss} -> [{synset_id, gloss} | acc]
        :skip -> acc
      end
    end)
    |> Enum.reverse()
  end

  defp parse_gloss_line(line) do
    case extract_args(line, "g") do
      nil ->
        :skip

      args_str ->
        case split_gloss_args(args_str) do
          {synset_id_str, gloss_str} ->
            gloss = unquote_prolog(gloss_str) |> extract_definition()
            {:ok, parse_int(synset_id_str), gloss}

          _ ->
            :skip
        end
    end
  end

  defp split_gloss_args(args_str) do
    case String.split(args_str, ",", parts: 2) do
      [synset_id_str, gloss_str] -> {String.trim(synset_id_str), String.trim(gloss_str)}
      _ -> nil
    end
  end

  defp extract_definition(gloss) do
    case String.split(gloss, "; \"", parts: 2) do
      [definition | _] -> String.trim(definition) |> String.trim_leading("(") |> String.trim_trailing(")")
      _ -> gloss
    end
  end

  @doc """
  Parses wn_hyp.pl (hypernym relations).

  Each line: hyp(synset_id,synset_id).

  Returns a list of `{child_synset_id, parent_synset_id}` tuples.
  """
  def parse_hypernyms(path) do
    parse_binary_relation(path, "hyp")
  end

  @doc """
  Parses wn_ant.pl (antonym relations).

  Each line: ant(synset_id,w_num,synset_id,w_num).

  Returns a list of `{synset_id1, w_num1, synset_id2, w_num2}` tuples.
  """
  def parse_antonyms(path) do
    path
    |> stream_lines()
    |> Enum.reduce([], fn line, acc ->
      case extract_args(line, "ant") do
        nil ->
          acc

        args_str ->
          case split_prolog_args(args_str) do
            [s1, w1, s2, w2] ->
              [{parse_int(s1), parse_int(w1), parse_int(s2), parse_int(w2)} | acc]

            _ ->
              acc
          end
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Parses wn_ins.pl (instance-of relations).

  Each line: ins(synset_id,synset_id).

  Returns a list of `{instance_synset_id, class_synset_id}` tuples.
  """
  def parse_instances(path) do
    parse_binary_relation(path, "ins")
  end

  @doc """
  Parses wn_sim.pl (similar-to relations for adjectives).

  Each line: sim(synset_id,synset_id).

  Returns a list of `{synset_id1, synset_id2}` tuples.
  """
  def parse_similar(path) do
    parse_binary_relation(path, "sim")
  end

  @doc """
  Parses wn_der.pl (derivationally related forms).

  Each line: der(synset_id,w_num,synset_id,w_num).

  Returns a list of `{synset_id1, w_num1, synset_id2, w_num2}` tuples.
  """
  def parse_derivations(path) do
    path
    |> stream_lines()
    |> Enum.reduce([], fn line, acc ->
      case extract_args(line, "der") do
        nil ->
          acc

        args_str ->
          case split_prolog_args(args_str) do
            [s1, w1, s2, w2] ->
              [{parse_int(s1), parse_int(w1), parse_int(s2), parse_int(w2)} | acc]

            _ ->
              acc
          end
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Parses wn_exc.pl (morphological exceptions).

  Each line: exc(pos,inflected_form,base_form).
  Example: exc(v,ran,run).

  Returns a list of `{inflected_form, base_form, pos}` tuples.
  """
  def parse_exceptions(path) do
    path
    |> stream_lines()
    |> Enum.reduce([], fn line, acc ->
      case extract_args(line, "exc") do
        nil ->
          acc

        args_str ->
          case split_prolog_args(args_str) do
            [pos, inflected, base] ->
              [{unquote_prolog(inflected), unquote_prolog(base),
                Map.get(@ss_type_map, pos, :unknown)} | acc]

            _ ->
              acc
          end
      end
    end)
    |> Enum.reverse()
  end

  # -- Internal helpers -------------------------------------------------------

  defp parse_binary_relation(path, operator) do
    path
    |> stream_lines()
    |> Enum.reduce([], fn line, acc ->
      case extract_args(line, operator) do
        nil ->
          acc

        args_str ->
          case split_prolog_args(args_str) do
            [s1, s2] -> [{parse_int(s1), parse_int(s2)} | acc]
            _ -> acc
          end
      end
    end)
    |> Enum.reverse()
  end

  defp stream_lines(path) do
    File.stream!(path, :line)
    |> Stream.map(&String.trim_trailing/1)
    |> Stream.reject(&(&1 == "" or String.starts_with?(&1, "%") or String.starts_with?(&1, ":-")))
  end

  @doc false
  def extract_args(line, operator) do
    prefix = operator <> "("

    if String.starts_with?(line, prefix) do
      line
      |> String.trim_trailing(".")
      |> String.slice(String.length(prefix)..-2//1)
    else
      nil
    end
  end

  @doc false
  def split_prolog_args(str) do
    do_split_args(str, [], "", false)
  end

  defp do_split_args("", acc, current, _in_quote) do
    Enum.reverse([current | acc])
  end

  defp do_split_args("''" <> rest, acc, current, true) do
    do_split_args(rest, acc, current <> "'", true)
  end

  defp do_split_args("'" <> rest, acc, current, false) do
    do_split_args(rest, acc, current, true)
  end

  defp do_split_args("'" <> rest, acc, current, true) do
    do_split_args(rest, acc, current, false)
  end

  defp do_split_args("," <> rest, acc, current, false) do
    do_split_args(rest, [current | acc], "", false)
  end

  defp do_split_args(<<char::utf8, rest::binary>>, acc, current, in_quote) do
    do_split_args(rest, acc, current <> <<char::utf8>>, in_quote)
  end

  defp unquote_prolog(str) do
    str
    |> String.trim_leading("'")
    |> String.trim_trailing("'")
    |> String.replace("''", "'")
  end

  defp parse_int(str) do
    case Integer.parse(String.trim(str)) do
      {n, _} -> n
      :error -> 0
    end
  end
end
