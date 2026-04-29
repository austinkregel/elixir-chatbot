defmodule Brain.ML.KnowledgeGraph.PredicateNormalizer do
  @moduledoc """
  Vocabulary alignment table for knowledge graph predicates.

  This module is **not** a string-matching classifier — it makes no routing
  or classification decisions. Its sole purpose is to map the heterogeneous
  predicate vocabularies used across the system (SRL verb lemmas, Epistemic
  atom predicates, ConceptNet relation types) onto the canonical PascalCase
  relation set that `Brain.ML.KnowledgeGraph.TripleScorer` was trained on.

  This is analogous to `Brain.ML.Gazetteer`, which maintains a static
  entity lookup table without performing entity classification.

  ## Predicate sources mapped

  - **SRL roles**: `LOCATED_AT`, `OCCURRED_AT`, `MANNER`, `CAUSED_BY`, `PURPOSE`,
    plus free-form verb predicates (`visited`, `wants`, etc.)
  - **Epistemic atoms**: `:likes`, `:wants`, `:interested_in`, `:needs`, etc.
  - **ConceptNet relations**: `IsA`, `AtLocation`, `Causes`, `HasProperty`, etc.
  - **Hierarchy/device predicates**: `is_a`, `has_subtype`, `made_by`, `located_in`

  ## Return values

  - `{:ok, canonical, :exact}` — input is already a canonical relation name
  - `{:ok, canonical, :mapped}` — input was found in the alias table
  - `{:ok, canonical, :synonym}` — case-insensitive match to a canonical name
  - `{:ok, canonical, :lemma_passthrough}` — verb lemma found in trained relation coverage
  - `{:error, :oov}` — out of vocabulary, consumers must not score this predicate
  - `{:error, :empty}` — empty or nil input
  """

  require Logger

  @aliases_path Path.join(:code.priv_dir(:brain) |> to_string(), "analysis/predicate_aliases.json")

  @type match_kind :: :exact | :mapped | :synonym | :lemma_passthrough
  @type normalize_result :: {:ok, String.t(), match_kind} | {:error, :oov | :empty}

  @doc """
  Normalize a predicate string or atom to the canonical relation name
  used by the TripleScorer.

  Returns `{:ok, canonical, kind}` on success, `{:error, reason}` otherwise.
  """
  @spec normalize(String.t() | atom() | nil) :: normalize_result
  def normalize(nil), do: {:error, :empty}
  def normalize(""), do: {:error, :empty}

  def normalize(predicate) when is_atom(predicate) do
    normalize(Atom.to_string(predicate))
  end

  def normalize(predicate) when is_binary(predicate) do
    trimmed = String.trim(predicate)
    if trimmed == "", do: {:error, :empty}, else: do_normalize(trimmed)
  end

  @doc """
  Returns all canonical relation names from the alias table.
  """
  @spec canonical_relations() :: [String.t()]
  def canonical_relations do
    alias_table()
    |> Map.keys()
    |> Enum.sort()
  end

  @doc """
  Returns the full alias table as `%{canonical => [aliases]}`.
  """
  @spec alias_table() :: %{String.t() => [String.t()]}
  def alias_table do
    case :persistent_term.get({__MODULE__, :alias_table}, nil) do
      nil ->
        table = load_alias_table()
        :persistent_term.put({__MODULE__, :alias_table}, table)
        table

      table ->
        table
    end
  end

  @doc """
  Force reload the alias table from disk. Useful after updating the JSON.
  """
  @spec reload!() :: :ok
  def reload! do
    table = load_alias_table()
    :persistent_term.put({__MODULE__, :alias_table}, table)
    :persistent_term.put({__MODULE__, :reverse_index}, build_reverse_index(table))
    :ok
  end

  defp do_normalize(predicate) do
    table = alias_table()
    reverse = reverse_index()

    cond do
      Map.has_key?(table, predicate) ->
        {:ok, predicate, :exact}

      Map.has_key?(reverse, predicate) ->
        {:ok, Map.fetch!(reverse, predicate), :mapped}

      (canonical = find_case_insensitive(predicate, table)) != nil ->
        {:ok, canonical, :synonym}

      true ->
        try_lemma_passthrough(predicate)
    end
  end

  defp reverse_index do
    case :persistent_term.get({__MODULE__, :reverse_index}, nil) do
      nil ->
        idx = build_reverse_index(alias_table())
        :persistent_term.put({__MODULE__, :reverse_index}, idx)
        idx

      idx ->
        idx
    end
  end

  defp build_reverse_index(table) do
    Enum.flat_map(table, fn {canonical, aliases} ->
      Enum.map(aliases, fn alias_str ->
        {alias_str, canonical}
      end)
    end)
    |> Map.new()
  end

  defp find_case_insensitive(predicate, table) do
    downcased = String.downcase(predicate)

    Enum.find_value(Map.keys(table), fn canonical ->
      if String.downcase(canonical) == downcased, do: canonical
    end)
  end

  defp try_lemma_passthrough(predicate) do
    lemma = Brain.ML.Lexicon.lemma(predicate)
    capitalized = String.capitalize(lemma)

    table = alias_table()

    cond do
      Map.has_key?(table, capitalized) ->
        {:ok, capitalized, :lemma_passthrough}

      relation_in_trained_coverage?(lemma) ->
        {:ok, capitalized, :lemma_passthrough}

      true ->
        {:error, :oov}
    end
  end

  defp relation_in_trained_coverage?(lemma) do
    case Brain.ML.KnowledgeGraph.TripleScorer.relation_coverage() do
      {:ok, coverage} when is_map(coverage) ->
        capitalized = String.capitalize(lemma)
        count = Map.get(coverage, capitalized, 0)
        count >= 5

      _ ->
        false
    end
  end

  defp load_alias_table do
    path = aliases_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, table} when is_map(table) -> table
          _ -> raise "Invalid predicate_aliases.json at #{path}: expected a JSON object"
        end

      {:error, reason} ->
        raise "Cannot load predicate_aliases.json at #{path}: #{inspect(reason)}"
    end
  end

  defp aliases_path do
    cond do
      File.exists?(@aliases_path) ->
        @aliases_path

      File.exists?("apps/brain/priv/analysis/predicate_aliases.json") ->
        "apps/brain/priv/analysis/predicate_aliases.json"

      true ->
        @aliases_path
    end
  end
end
