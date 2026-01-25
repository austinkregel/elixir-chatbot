defmodule ChatBot.Analysis.IntentRegistry do
  @moduledoc """
  Centralized registry for intent metadata.
  Replaces scattered keyword-based intent checks with structured lookups.

  This module provides a single source of truth for intent properties including:
  - Domain (weather, music, device, navigation, smalltalk, etc.)
  - Category (expressive, directive, assertive)
  - Speech act type (greeting, farewell, command, request_information, etc.)
  - Required and optional entities
  - Entity mappings and clarification templates
  """

  # Load registry at compile time
  @registry_path "priv/analysis/intent_registry.json"
  @external_resource @registry_path

  @registry (case File.read(@registry_path) do
               {:ok, content} ->
                 case Jason.decode(content) do
                   {:ok, data} -> data
                   {:error, _} -> %{}
                 end

               {:error, _} ->
                 %{}
             end)

  @doc """
  Get full metadata for an intent.
  Returns nil if intent is not in registry.
  """
  def get(nil), do: nil
  def get(""), do: nil

  def get(intent) when is_binary(intent) do
    Map.get(@registry, intent)
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

  @doc "Get entity mappings for an intent."
  def entity_mappings(intent) do
    case get(intent) do
      nil -> %{}
      meta -> meta["entity_mappings"] || %{}
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
    Map.keys(@registry)
  end

  @doc "List all intents for a given domain."
  def list_by_domain(domain) when is_atom(domain) do
    domain_str = to_string(domain)

    @registry
    |> Enum.filter(fn {_intent, meta} -> meta["domain"] == domain_str end)
    |> Enum.map(fn {intent, _meta} -> intent end)
  end

  @doc "List all intents for a given category."
  def list_by_category(category) when is_atom(category) do
    category_str = to_string(category)

    @registry
    |> Enum.filter(fn {_intent, meta} -> meta["category"] == category_str end)
    |> Enum.map(fn {intent, _meta} -> intent end)
  end

  # Private helpers

  defp to_atom_or_nil(nil), do: nil
  defp to_atom_or_nil(""), do: nil
  defp to_atom_or_nil(str) when is_binary(str), do: String.to_atom(str)
  defp to_atom_or_nil(atom) when is_atom(atom), do: atom
end
