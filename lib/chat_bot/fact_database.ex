defmodule ChatBot.FactDatabase do
  @moduledoc """
  Fact database for storing and querying verifiable general knowledge facts.

  This module provides access to a curated database of true, verifiable facts
  that can be used for testing and knowledge building. Facts are organized by
  category and include verification sources.

  Facts are stored in JSON files under `data/facts/` and loaded at runtime.
  All facts are returned as `ChatBot.FactDatabase.Fact` structs.
  """

  use GenServer
  require Logger

  alias ChatBot.FactDatabase.Fact

  @facts_dir "data/facts"

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Query facts by category, entity, or keyword search.

  Options:
  - `:category` - Filter by category (e.g., "geography", "science")
  - `:entity` - Filter by entity name (e.g., "France", "water")
  - `:search` - Search in fact text
  - `:limit` - Maximum number of results (default: 10)
  """
  def query(opts \\ []) do
    GenServer.call(__MODULE__, {:query, opts})
  end

  @doc """
  Get a specific fact by ID.
  """
  def get_fact(id) do
    GenServer.call(__MODULE__, {:get_fact, id})
  end

  @doc """
  Get all facts for a specific entity.
  """
  def get_entity_facts(entity_name) do
    GenServer.call(__MODULE__, {:get_entity_facts, entity_name})
  end

  @doc """
  Get all facts in a category.
  """
  def get_category_facts(category) do
    GenServer.call(__MODULE__, {:get_category_facts, category})
  end

  @doc """
  Get all available categories.
  """
  def list_categories do
    GenServer.call(__MODULE__, :list_categories)
  end

  @doc """
  Reload facts from disk.
  """
  def reload do
    GenServer.call(__MODULE__, :reload)
  end

  @doc """
  Add a fact dynamically (for learned facts).
  This is a low-level function - use FactDatabase.Integration.add_fact/3 for full integration.
  """
  def add_fact_direct(fact_map) when is_map(fact_map) do
    GenServer.call(__MODULE__, {:add_fact_direct, fact_map})
  end

  @doc """
  Get statistics about the fact database.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Checks if the fact database is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    facts = load_all_facts()
    Logger.info("FactDatabase started", %{fact_count: length(facts)})
    {:ok, %{facts: facts, loaded_at: System.system_time(:second)}}
  end

  @impl true
  def handle_call({:query, opts}, _from, state) do
    results =
      state.facts
      |> filter_by_category(opts[:category])
      |> filter_by_entity(opts[:entity])
      |> search_in_text(opts[:search])
      |> limit_results(opts[:limit] || 10)

    {:reply, results, state}
  end

  @impl true
  def handle_call({:get_fact, id}, _from, state) do
    fact = Enum.find(state.facts, &(&1.id == id))
    {:reply, fact, state}
  end

  @impl true
  def handle_call({:get_entity_facts, entity_name}, _from, state) do
    normalized = String.downcase(entity_name)

    facts =
      Enum.filter(state.facts, fn fact ->
        String.downcase(fact.entity) == normalized
      end)

    {:reply, facts, state}
  end

  @impl true
  def handle_call({:get_category_facts, category}, _from, state) do
    normalized = String.downcase(category)

    facts =
      Enum.filter(state.facts, fn fact ->
        String.downcase(fact.category) == normalized
      end)

    {:reply, facts, state}
  end

  @impl true
  def handle_call(:list_categories, _from, state) do
    categories =
      state.facts
      |> Enum.map(& &1.category)
      |> Enum.uniq()
      |> Enum.sort()

    {:reply, categories, state}
  end

  @impl true
  def handle_call(:reload, _from, _state) do
    facts = load_all_facts()
    Logger.info("FactDatabase reloaded", %{fact_count: length(facts)})
    {:reply, :ok, %{facts: facts, loaded_at: System.system_time(:second)}}
  end

  @impl true
  def handle_call({:add_fact_direct, fact_map}, _from, state) do
    # Convert map to Fact struct (handles field normalization)
    fact = Fact.from_map(fact_map)

    updated_facts = [fact | state.facts]
    new_state = %{state | facts: updated_facts}

    Logger.debug("Added fact directly", %{fact_id: fact.id})
    {:reply, {:ok, fact.id}, new_state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      total_facts: length(state.facts),
      categories: state.facts |> Enum.map(& &1.category) |> Enum.uniq() |> length(),
      entities: state.facts |> Enum.map(& &1.entity) |> Enum.uniq() |> length(),
      loaded_at: state.loaded_at
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # Private Functions

  defp load_all_facts do
    facts_dir = Path.join([File.cwd!(), @facts_dir])

    if File.exists?(facts_dir) do
      facts_dir
      |> Path.join("*.json")
      |> Path.wildcard()
      |> Enum.flat_map(&load_facts_file/1)
    else
      Logger.warning("Facts directory not found", %{path: facts_dir})
      []
    end
  end

  defp load_facts_file(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            facts =
              data
              |> Map.get("facts", [])
              |> Enum.map(&Fact.from_map/1)

            Logger.debug("Loaded facts from file", %{
              file: Path.basename(file_path),
              count: length(facts)
            })

            facts

          {:error, reason} ->
            Logger.error("Failed to parse facts file", %{file: file_path, reason: reason})
            []
        end

      {:error, reason} ->
        Logger.error("Failed to read facts file", %{file: file_path, reason: reason})
        []
    end
  end

  defp filter_by_category(facts, nil), do: facts

  defp filter_by_category(facts, category) when is_binary(category) do
    normalized = String.downcase(category)

    Enum.filter(facts, fn fact ->
      String.downcase(fact.category) == normalized
    end)
  end

  defp filter_by_category(facts, _), do: facts

  defp filter_by_entity(facts, nil), do: facts

  defp filter_by_entity(facts, entity) when is_binary(entity) do
    normalized = String.downcase(entity)

    Enum.filter(facts, fn fact ->
      String.downcase(fact.entity) == normalized
    end)
  end

  defp filter_by_entity(facts, _), do: facts

  defp search_in_text(facts, nil), do: facts

  defp search_in_text(facts, search_term) when is_binary(search_term) do
    # Extract meaningful keywords (skip common words)
    keywords = extract_keywords(search_term)

    # Need at least one keyword to search
    if keywords == [] do
      []
    else
      facts
      |> Enum.map(fn fact ->
        fact_text = String.downcase(fact.fact)
        entity_text = String.downcase(fact.entity)

        # Count matches in entity (weighted more heavily) and fact text
        entity_matches = Enum.count(keywords, &String.contains?(entity_text, &1))
        fact_matches = Enum.count(keywords, &String.contains?(fact_text, &1))

        # Calculate relevance score:
        # - Entity match is worth 2 points (entity is the topic of the fact)
        # - Fact text match is worth 1 point
        score = entity_matches * 2 + fact_matches

        {fact, score, entity_matches}
      end)
      |> Enum.filter(fn {_fact, score, entity_matches} ->
        # Require either:
        # - At least one entity match, OR
        # - Score of 3+ (multiple content word matches in the fact)
        entity_matches >= 1 or score >= 3
      end)
      |> Enum.sort_by(fn {_fact, score, _} -> -score end)
      |> Enum.map(fn {fact, _, _} -> fact end)
    end
  end

  defp search_in_text(facts, _), do: facts

  # Extract meaningful keywords from search text using POS tagging
  # Content words (NOUN, PROPN, VERB, ADJ, ADV, NUM) are meaningful for search
  defp extract_keywords(text) do
    alias ChatBot.ML.Tokenizer
    alias ChatBot.ML.POSTagger

    # Content POS tags that indicate meaningful search terms
    content_tags = ~w(NOUN PROPN VERB ADJ ADV NUM)

    tokens = Tokenizer.tokenize(text)
    token_texts = Enum.map(tokens, fn t -> t.text end)

    case POSTagger.load_model() do
      {:ok, model} ->
        POSTagger.predict(token_texts, model)
        |> Enum.filter(fn {_word, tag} -> tag in content_tags end)
        |> Enum.map(fn {word, _tag} -> String.downcase(word) end)
        |> Enum.filter(fn w -> String.length(w) > 2 end)

      {:error, _} ->
        # Fallback: use word tokens with length > 2
        tokens
        |> Enum.filter(fn t ->
          t.type in [:word, :number] and String.length(t.text) > 2
        end)
        |> Enum.map(fn t -> String.downcase(t.text) end)
    end
  end

  defp limit_results(facts, limit) when is_integer(limit) and limit > 0 do
    Enum.take(facts, limit)
  end

  defp limit_results(facts, _), do: facts
end
