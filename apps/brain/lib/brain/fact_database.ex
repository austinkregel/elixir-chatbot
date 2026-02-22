defmodule Brain.FactDatabase do
  @moduledoc """
  Fact database for storing and querying verifiable general knowledge facts.

  This module manages two distinct layers of facts:

  ## Curated Facts (Immutable)

  Curated facts are grounding truths loaded from JSON files in `data/facts/`.
  They represent agreed-upon reality — verifiable facts with authoritative sources.
  These facts are **read-only**: no API can modify, delete, or overwrite them.
  When registered with the epistemic system, they carry the `:curated_fact`
  source authority (confidence 1.0, JTMS premise, no decay).

  ## Learned Facts (Mutable)

  Learned facts are acquired dynamically through conversation and research,
  persisted to Atlas. They can be added, modified, and carry lower authority
  levels based on their source.

  All facts are returned as `Brain.FactDatabase.Fact` structs. Queries search
  across both layers transparently.
  """

  use GenServer
  require Logger

  alias Brain.FactDatabase.Fact

  @default_facts_dir "data/facts"

  defp facts_dir do
    Application.get_env(:brain, :facts_dir, @default_facts_dir)
  end


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
  - `:layer` - Filter by layer: `:curated`, `:learned`, or `:all` (default: `:all`)
  """
  def query(opts \\ []) do
    Brain.Telemetry.span(:fact_database_query, %{opts: opts}, fn ->
      GenServer.call(__MODULE__, {:query, opts}, 5_000)
    end)
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
  Reload facts from disk and Atlas.

  Curated facts are re-read from the JSON files (immutable source of truth).
  Learned facts are re-loaded from Atlas.
  """
  def reload do
    GenServer.call(__MODULE__, :reload)
  end

  @doc """
  Add a learned fact dynamically.

  This is a low-level function — use `FactDatabase.Integration.add_fact/3` for
  full epistemic integration. The fact is added to the **learned** layer only.
  Attempting to add a fact with an ID that matches a curated fact will be rejected.
  """
  def add_fact_direct(fact_map) when is_map(fact_map) do
    GenServer.call(__MODULE__, {:add_fact_direct, fact_map})
  end

  @doc """
  Check whether a fact ID belongs to the curated (immutable) layer.
  """
  def curated?(fact_id) do
    GenServer.call(__MODULE__, {:curated?, fact_id})
  end

  @doc """
  Get statistics about the fact database, broken down by layer.
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
    {curated, learned} = load_all_facts_from_atlas()
    curated_ids = MapSet.new(curated, & &1.id)

    Logger.info("FactDatabase started", %{
      curated_count: length(curated),
      learned_count: length(learned)
    })

    {:ok, %{
      curated: curated,
      curated_ids: curated_ids,
      learned: learned,
      loaded_at: System.system_time(:second)
    }}
  end

  @impl true
  def handle_call({:query, opts}, _from, state) do
    layer = Keyword.get(opts, :layer, :all)

    facts = select_layer(state, layer)

    results =
      facts
      |> filter_by_category(opts[:category])
      |> filter_by_entity(opts[:entity])
      |> search_in_text(opts[:search])
      |> limit_results(opts[:limit] || 10)

    {:reply, results, state}
  end

  @impl true
  def handle_call({:get_fact, id}, _from, state) do
    fact = Enum.find(all_facts(state), &(&1.id == id))
    {:reply, fact, state}
  end

  @impl true
  def handle_call({:get_entity_facts, entity_name}, _from, state) do
    normalized = String.downcase(entity_name)

    facts =
      Enum.filter(all_facts(state), fn fact ->
        String.downcase(fact.entity) == normalized
      end)

    {:reply, facts, state}
  end

  @impl true
  def handle_call({:get_category_facts, category}, _from, state) do
    normalized = String.downcase(category)

    facts =
      Enum.filter(all_facts(state), fn fact ->
        String.downcase(fact.category) == normalized
      end)

    {:reply, facts, state}
  end

  @impl true
  def handle_call(:list_categories, _from, state) do
    categories =
      all_facts(state)
      |> Enum.map(& &1.category)
      |> Enum.uniq()
      |> Enum.sort()

    {:reply, categories, state}
  end

  @impl true
  def handle_call(:reload, _from, _state) do
    {curated, learned} = load_all_facts_from_atlas()
    curated_ids = MapSet.new(curated, & &1.id)

    Logger.info("FactDatabase reloaded", %{
      curated_count: length(curated),
      learned_count: length(learned)
    })

    {:reply, :ok, %{
      curated: curated,
      curated_ids: curated_ids,
      learned: learned,
      loaded_at: System.system_time(:second)
    }}
  end

  @impl true
  def handle_call({:add_fact_direct, fact_map}, _from, state) do
    fact = Fact.from_map(fact_map)

    if MapSet.member?(state.curated_ids, fact.id) do
      Logger.warning("Rejected attempt to overwrite curated fact", %{fact_id: fact.id})
      {:reply, {:error, :curated_fact_immutable}, state}
    else
      updated_learned = [fact | state.learned]
      new_state = %{state | learned: updated_learned}

      Logger.debug("Added learned fact", %{fact_id: fact.id})
      {:reply, {:ok, fact.id}, new_state}
    end
  end

  @impl true
  def handle_call({:curated?, fact_id}, _from, state) do
    {:reply, MapSet.member?(state.curated_ids, fact_id), state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    all = all_facts(state)

    stats = %{
      total_facts: length(all),
      curated_facts: length(state.curated),
      learned_facts: length(state.learned),
      categories: all |> Enum.map(& &1.category) |> Enum.uniq() |> length(),
      entities: all |> Enum.map(& &1.entity) |> Enum.uniq() |> length(),
      loaded_at: state.loaded_at
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # Private Functions — Layer helpers

  defp all_facts(state), do: state.curated ++ state.learned

  defp select_layer(state, :curated), do: state.curated
  defp select_layer(state, :learned), do: state.learned
  defp select_layer(state, _), do: all_facts(state)

  # Private Functions — Loading
  #
  # All facts (curated + learned) live in atlas_learned_facts.
  # The importer seeds them from data/facts/*.json.
  # We partition into curated vs learned based on whether the fact
  # has a learned_at timestamp — curated facts don't.

  defp load_all_facts_from_atlas do
    case Brain.AtlasIntegration.load_learned_facts() do
      {:ok, facts} ->
        {curated, learned} = Enum.split_with(facts, &curated_fact?/1)

        Logger.debug("Loaded facts from Atlas", %{
          curated: length(curated),
          learned: length(learned)
        })

        {curated, learned}

      {:error, reason} ->
        Logger.warning("Failed to load facts from Atlas, falling back to files",
          reason: inspect(reason)
        )
        {load_curated_facts_from_files(), []}
    end
  rescue
    e ->
      Logger.warning("Atlas unavailable for facts: #{inspect(e)}")
      {load_curated_facts_from_files(), []}
  end

  # Curated facts have no learned_at timestamp and come from known
  # categories (geography, science, history, general)
  @curated_categories MapSet.new(~w(geography science history general))

  defp curated_fact?(%Fact{learned_at: nil, category: cat}) do
    MapSet.member?(@curated_categories, String.downcase(cat))
  end

  defp curated_fact?(_), do: false

  defp load_curated_facts_from_files do
    base_dir = facts_dir()

    facts_dir_path =
      if Path.type(base_dir) == :absolute do
        base_dir
      else
        Path.join([File.cwd!(), base_dir])
      end

    if File.exists?(facts_dir_path) do
      facts_dir_path
      |> Path.join("*.json")
      |> Path.wildcard()
      |> Enum.flat_map(&load_facts_file/1)
    else
      Logger.debug("No curated facts available (Atlas empty, no file fallback)",
        %{path: facts_dir_path}
      )
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
    alias Brain.ML.Tokenizer
    alias Brain.ML.POSTagger

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
