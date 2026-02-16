defmodule Brain.Analysis.IntentRegistry do
  @moduledoc """
  Centralized registry for intent metadata, backed by GenServer + ETS.

  Provides a single source of truth for intent properties including:
  - Domain (weather, music, device, navigation, smalltalk, etc.)
  - Category (expressive, directive, assertive)
  - Speech act type (greeting, farewell, command, request_information, etc.)
  - Required and optional entities
  - Entity mappings and clarification templates

  At runtime, data is served from ETS for fast concurrent reads.
  A compile-time fallback registry is kept for cases where the GenServer
  is not yet started (compilation, tests, startup race).

  New intents can be registered at runtime via `register_intent/2`.
  """

  use GenServer
  require Logger

  @ets_table :intent_registry
  @registry_path "priv/analysis/intent_registry.json"
  @external_resource @registry_path

  # Compile-time fallback loaded from the same JSON
  @fallback_registry (case File.read(@registry_path) do
                        {:ok, content} ->
                          case Jason.decode(content) do
                            {:ok, data} -> data
                            {:error, _} -> %{}
                          end

                        {:error, _} ->
                          %{}
                      end)

  # Speech act to intent mapping (static, no need for ETS)
  @speech_act_intent_map %{
    greeting: "smalltalk.greetings.hello",
    farewell: "smalltalk.greetings.bye",
    thanks: "smalltalk.appraisal.thank_you",
    apology: "smalltalk.dialog.sorry",
    how_are_you: "smalltalk.greetings.how_are_you",
    compliment: "smalltalk.appraisal.good",
    backchannel: "smalltalk.confirmation.ok",
    continuation: "smalltalk.dialog.continue"
  }

  # --- GenServer lifecycle ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @impl true
  def init(opts) do
    table = :ets.new(@ets_table, [:named_table, :set, :public, read_concurrency: true])

    # Load from JSON on init
    registry = load_registry(opts)

    for {intent, meta} <- registry do
      :ets.insert(table, {intent, meta})
    end

    Logger.info("IntentRegistry loaded #{map_size(registry)} intents into ETS")

    {:ok, %{registry_path: registry_json_path(opts)}}
  end

  # --- Core lookup helper ---

  # Single internal helper wrapping ETS lookup with fallback
  defp get_entry(intent) when is_binary(intent) do
    if :ets.whereis(@ets_table) != :undefined do
      case :ets.lookup(@ets_table, intent) do
        [{^intent, meta}] -> meta
        [] -> nil
      end
    else
      # Fallback to compile-time data when ETS not available
      Map.get(@fallback_registry, intent)
    end
  rescue
    ArgumentError ->
      Map.get(@fallback_registry, intent)
  end

  defp get_entry(_), do: nil

  # Helper to get all entries from ETS or fallback
  defp all_entries do
    if :ets.whereis(@ets_table) != :undefined do
      :ets.tab2list(@ets_table) |> Map.new()
    else
      @fallback_registry
    end
  rescue
    ArgumentError -> @fallback_registry
  end

  # --- Public API ---

  @doc """
  Get full metadata for an intent.
  Returns nil if intent is not in registry.
  """
  def get(nil), do: nil
  def get(""), do: nil

  def get(intent) when is_binary(intent) do
    get_entry(intent)
  end

  def get(intent) when is_atom(intent), do: get(to_string(intent))

  @doc """
  Get the domain for an intent.
  Returns atom like :weather, :music, :device, :smalltalk, etc.
  """
  def domain(intent) do
    case get(intent) do
      nil -> nil
      meta -> to_atom_or_nil(meta["domain"])
    end
  end

  @doc """
  Get the category for an intent.
  Returns :expressive, :directive, or :assertive.
  """
  def category(intent) do
    case get(intent) do
      nil -> nil
      meta -> to_atom_or_nil(meta["category"])
    end
  end

  @doc """
  Get the speech act type for an intent.
  Returns atom like :greeting, :farewell, :command, :request_information, etc.
  """
  def speech_act(intent) do
    case get(intent) do
      nil -> nil
      meta -> to_atom_or_nil(meta["speech_act"])
    end
  end

  @doc """
  Get the query type for meta-cognitive intents.
  Returns atom like :self_query, :memory_check, :privacy_probe, :trust_check.
  """
  def query_type(intent) do
    case get(intent) do
      nil -> nil
      meta -> to_atom_or_nil(meta["query_type"])
    end
  end

  # Domain predicates

  @doc "Returns true if intent is a weather-related intent."
  def weather_intent?(intent), do: domain(intent) == :weather

  @doc "Returns true if intent is a music-related intent."
  def music_intent?(intent), do: domain(intent) == :music

  @doc "Returns true if intent is a navigation-related intent."
  def navigation_intent?(intent), do: domain(intent) == :navigation

  @doc "Returns true if intent is a device control intent."
  def device_intent?(intent), do: domain(intent) == :device

  @doc "Returns true if intent is a smalltalk intent."
  def smalltalk_intent?(intent), do: domain(intent) == :smalltalk

  @doc "Returns true if intent is a search intent."
  def search_intent?(intent), do: domain(intent) == :search

  @doc "Returns true if intent is a meta-cognitive intent."
  def meta_intent?(intent), do: domain(intent) == :meta

  @doc "Returns true if intent is an introduction/self-identification intent."
  def introduction_intent?(intent), do: domain(intent) == :introduction

  # Entity mapping functions

  @doc """
  Returns list of entity types expected by an intent based on entity_mappings.

  This is used by EntityDisambiguator to dynamically determine which entity
  types should be preferred for a given intent, rather than hardcoding
  context preferences for each domain.

  ## Examples

      iex> IntentRegistry.expected_entity_types("smarthome.device.switch.on")
      ["device", "room"]

      iex> IntentRegistry.expected_entity_types("weather.query")
      ["location", "room", "city", "ambiguous_name_location", "date", "relative_date", "sys-date", "time", "sys-time", "unit-temperature"]

      iex> IntentRegistry.expected_entity_types("unknown")
      []
  """
  def expected_entity_types(intent) do
    case get(intent) do
      nil ->
        []

      meta ->
        meta
        |> Map.get("entity_mappings", %{})
        |> Map.values()
        |> List.flatten()
        |> Enum.uniq()
    end
  end

  @doc """
  Returns a map of slot names to their expected entity types for an intent.

  This provides more granular information than expected_entity_types/1,
  preserving the relationship between slots and entity types.
  """
  def entity_mappings(intent) do
    case get(intent) do
      nil -> %{}
      meta -> Map.get(meta, "entity_mappings", %{})
    end
  end

  # Category predicates

  @doc "Returns true if intent is expressive (greeting, farewell, thanks, etc.)."
  def expressive?(intent), do: category(intent) == :expressive

  @doc "Returns true if intent is directive (command, request)."
  def directive?(intent), do: category(intent) == :directive

  @doc "Returns true if intent is assertive (statement, assertion)."
  def assertive?(intent), do: category(intent) == :assertive

  # Speech act predicates

  @doc "Returns true if intent is a greeting."
  def greeting?(intent), do: speech_act(intent) == :greeting

  @doc "Returns true if intent is a farewell."
  def farewell?(intent), do: speech_act(intent) == :farewell

  @doc "Returns true if intent is a thanks expression."
  def thanks?(intent), do: speech_act(intent) == :thanks

  @doc "Returns true if intent is an apology."
  def apology?(intent), do: speech_act(intent) == :apology

  @doc "Returns true if intent is a command."
  def command?(intent), do: speech_act(intent) == :command

  @doc "Returns true if intent is a request for information."
  def request_information?(intent), do: speech_act(intent) == :request_information

  @doc "Returns true if intent is a backchannel."
  def backchannel?(intent), do: speech_act(intent) == :backchannel

  @doc "Returns true if intent is a continuation."
  def continuation?(intent), do: speech_act(intent) == :continuation

  # Speech act to intent mapping

  @doc """
  Get the canonical intent name for a speech act sub_type.
  Used to map expressive speech acts to their template intents.

  ## Examples

      iex> IntentRegistry.intent_for_speech_act(:greeting)
      "smalltalk.greetings.hello"

      iex> IntentRegistry.intent_for_speech_act(:unknown)
      nil
  """
  def intent_for_speech_act(sub_type) when is_atom(sub_type) do
    Map.get(@speech_act_intent_map, sub_type)
  end

  def intent_for_speech_act(_), do: nil

  @doc """
  Get all registered speech act to intent mappings.
  """
  def speech_act_intent_mappings, do: @speech_act_intent_map

  # Specificity check

  @doc """
  Returns true if intent is specific (not smalltalk, unknown, or nil).
  Used to determine if an intent warrants phrase matching or specific handling.
  """
  def specific?(intent) do
    d = domain(intent)
    d != nil and d not in [:smalltalk, :unknown]
  end

  # Entity requirements

  @doc "Get required entities for an intent."
  def required_entities(intent) do
    case get(intent) do
      nil -> []
      meta -> meta["required"] || []
    end
  end

  @doc "Get optional entities for an intent."
  def optional_entities(intent) do
    case get(intent) do
      nil -> []
      meta -> meta["optional"] || []
    end
  end

  @doc "Get clarification templates for an intent."
  def clarification_templates(intent) do
    case get(intent) do
      nil -> %{}
      meta -> meta["clarification_templates"] || %{}
    end
  end

  @doc "Get default values for an intent."
  def defaults(intent) do
    case get(intent) do
      nil -> %{}
      meta -> meta["defaults"] || %{}
    end
  end

  @doc "Get description for an intent."
  def description(intent) do
    case get(intent) do
      nil -> nil
      meta -> meta["description"]
    end
  end

  @doc "List all registered intents."
  def list_intents do
    all_entries() |> Map.keys()
  end

  @doc "List all intents for a given domain."
  def list_by_domain(domain) when is_atom(domain) do
    domain_str = to_string(domain)

    all_entries()
    |> Enum.filter(fn {_intent, meta} -> meta["domain"] == domain_str end)
    |> Enum.map(fn {intent, _meta} -> intent end)
  end

  @doc "List all intents for a given category."
  def list_by_category(category) when is_atom(category) do
    category_str = to_string(category)

    all_entries()
    |> Enum.filter(fn {_intent, meta} -> meta["category"] == category_str end)
    |> Enum.map(fn {intent, _meta} -> intent end)
  end

  @doc "Get enrichment sources for an intent. Returns a list like [\"system_stats\"]."
  def enrichment_sources(intent) do
    case get(intent) do
      nil -> []
      meta -> meta["enrichment_sources"] || []
    end
  end

  @doc "List all intents that require a given enrichment source."
  def intents_with_enrichment_source(source) when is_binary(source) do
    all_entries()
    |> Enum.filter(fn {_intent, meta} ->
      source in (meta["enrichment_sources"] || [])
    end)
    |> Enum.map(fn {intent, _meta} -> intent end)
  end

  @doc """
  Converts intent name to human-readable format.

  ## Examples

      iex> IntentRegistry.humanize("weather.query")
      "weather query"

      iex> IntentRegistry.humanize("smalltalk.greetings.hello")
      "greetings hello"

      iex> IntentRegistry.humanize(nil)
      "something"
  """
  def humanize(nil), do: "something"
  def humanize(""), do: "something"

  def humanize(intent) when is_binary(intent) do
    intent
    |> String.replace(".", " ")
    |> String.replace("_", " ")
    |> String.replace("smalltalk ", "")
    |> String.trim()
  end

  def humanize(_), do: "something"

  # --- Runtime registration ---

  @doc """
  Register a new intent at runtime. Persists to JSON for restart durability.
  """
  def register_intent(intent, metadata, name \\ __MODULE__)
      when is_binary(intent) and is_map(metadata) do
    GenServer.call(name, {:register_intent, intent, metadata})
  end

  @doc """
  Checks if the GenServer is ready.
  """
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Reload the registry from JSON file.
  """
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload)
  end

  # --- GenServer callbacks ---

  @impl true
  def handle_call({:register_intent, intent, metadata}, _from, state) do
    :ets.insert(@ets_table, {intent, metadata})

    # Persist to JSON
    persist_registry(state.registry_path)

    Logger.info("IntentRegistry: registered new intent '#{intent}'")
    {:reply, :ok, state}
  end

  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  def handle_call(:reload, _from, state) do
    registry = load_registry_from_path(state.registry_path)

    :ets.delete_all_objects(@ets_table)

    for {intent, meta} <- registry do
      :ets.insert(@ets_table, {intent, meta})
    end

    Logger.info("IntentRegistry reloaded #{map_size(registry)} intents")
    {:reply, :ok, state}
  end

  # --- Private helpers ---

  defp to_atom_or_nil(nil), do: nil
  defp to_atom_or_nil(""), do: nil
  defp to_atom_or_nil(str) when is_binary(str) do
    String.to_existing_atom(str)
  rescue
    ArgumentError -> nil
  end
  defp to_atom_or_nil(atom) when is_atom(atom), do: atom

  defp load_registry(opts) do
    path = registry_json_path(opts)
    load_registry_from_path(path)
  end

  defp load_registry_from_path(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) -> data
          _ -> @fallback_registry
        end

      {:error, _} ->
        @fallback_registry
    end
  end

  defp registry_json_path(opts) do
    case Keyword.get(opts, :registry_path) do
      nil ->
        case :code.priv_dir(:brain) do
          {:error, _} -> @registry_path
          priv_dir -> Path.join(priv_dir, "analysis/intent_registry.json")
        end

      path ->
        path
    end
  end

  defp persist_registry(path) do
    all = all_entries()
    json = Jason.encode!(all, pretty: true)
    dir = Path.dirname(path)
    File.mkdir_p!(dir)
    File.write!(path, json)
  rescue
    e ->
      Logger.warning("Failed to persist intent registry: #{Exception.message(e)}")
  end
end
