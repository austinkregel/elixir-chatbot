defmodule Brain.Lexicon.ConceptNetParser do
  @moduledoc """
  Parses ConceptNet 5 CSV assertion files into the canonical lexicon format.

  ConceptNet assertions are TSV with columns:
  URI, relation, start, end, metadata_json

  We extract English-only assertions and group them by concept
  into `%{concept => %{relation_type => [related_concepts]}}`.
  """

  require Logger

  @english_prefix "/c/en/"

  @kept_relations ~w(
    IsA PartOf HasA UsedFor CapableOf AtLocation Causes
    HasProperty MotivatedByGoal CreatedBy DefinedAs SymbolOf
    MadeOf ReceivesAction HasPrerequisite HasSubevent
    HasFirstSubevent HasLastSubevent CausesDesire
    DesireOf Desires NotDesires
  )

  @doc """
  Parses a ConceptNet CSV/TSV file and returns a map of
  `%{concept_string => %{relation_type => [related_concept_strings]}}`.

  Only keeps English concepts and the relation types in @kept_relations.
  """
  def parse(path, opts \\ []) do
    max_entries = Keyword.get(opts, :max_entries, :infinity)
    min_weight = Keyword.get(opts, :min_weight, 1.0)

    Logger.info("ConceptNetParser: parsing #{path}...")

    result =
      path
      |> File.stream!(:line)
      |> Stream.reject(&String.starts_with?(&1, "#"))
      |> Stream.map(&parse_line/1)
      |> Stream.reject(&is_nil/1)
      |> Stream.filter(fn {_rel, _start_c, _end_c, weight} -> weight >= min_weight end)
      |> maybe_take(max_entries)
      |> Enum.reduce(%{}, fn {relation, start_concept, end_concept, _weight}, acc ->
        acc
        |> add_relation(start_concept, relation, end_concept)
        |> add_relation(end_concept, relation <> "_inverse", start_concept)
      end)

    Logger.info("ConceptNetParser: extracted #{map_size(result)} concepts")
    result
  end

  @doc """
  Writes parsed ConceptNet data to a .term file for fast loading.
  """
  def parse_and_save(input_path, output_path, opts \\ []) do
    data = parse(input_path, opts)
    File.mkdir_p!(Path.dirname(output_path))
    binary = :erlang.term_to_binary(data, [:compressed])
    File.write!(output_path, binary)
    Logger.info("ConceptNetParser: saved #{map_size(data)} concepts to #{output_path}")
    {:ok, map_size(data)}
  end

  # -- Private ----------------------------------------------------------------

  defp parse_line(line) do
    case String.split(String.trim(line), "\t") do
      [_uri, relation_uri, start_uri, end_uri, metadata | _] ->
        relation = extract_relation(relation_uri)
        start_concept = extract_english_concept(start_uri)
        end_concept = extract_english_concept(end_uri)
        weight = extract_weight(metadata)

        if relation && start_concept && end_concept && relation in @kept_relations do
          {relation, start_concept, end_concept, weight}
        else
          nil
        end

      _ ->
        nil
    end
  end

  defp extract_relation(uri) do
    case String.split(uri, "/") do
      ["", "r", relation | _] -> relation
      _ -> nil
    end
  end

  defp extract_english_concept(uri) do
    if String.starts_with?(uri, @english_prefix) do
      uri
      |> String.trim_leading(@english_prefix)
      |> String.split("/")
      |> List.first()
      |> String.replace("_", " ")
    else
      nil
    end
  end

  defp extract_weight(metadata_str) do
    case Jason.decode(metadata_str) do
      {:ok, %{"weight" => weight}} when is_number(weight) -> weight
      _ -> 1.0
    end
  rescue
    _ -> 1.0
  end

  defp add_relation(acc, concept, relation, related) do
    Map.update(acc, concept, %{relation => [related]}, fn rels ->
      Map.update(rels, relation, [related], fn existing ->
        [related | existing] |> Enum.uniq() |> Enum.take(50)
      end)
    end)
  end

  defp maybe_take(stream, :infinity), do: stream
  defp maybe_take(stream, n), do: Stream.take(stream, n)
end
