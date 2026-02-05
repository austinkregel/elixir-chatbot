defmodule Brain.Analysis.EventPatterns do
  @moduledoc """
  Loads and provides access to event patterns from priv/analysis/event_patterns.json.

  Event patterns define POS tag sequences that represent common syntactic structures
  for extracting actor-verb-object relationships. Patterns are compiled at build time
  and converted to tensor-compatible indices for GPU-accelerated matching.

  ## Pattern Structure

  Each pattern has:
  - `structure`: List of POS tags (e.g., ["PRON", "VERB", "NOUN"])
  - `roles`: Map of semantic roles to position indices
  - `weight`: Pattern confidence weight (0.0-1.0)
  - `tense_hint`: Optional tense indicator for imperative patterns

  ## Usage

      EventPatterns.patterns()
      # => [%{id: "svo_basic", structure: ["PRON", "VERB", "NOUN"], ...}]

      EventPatterns.pos_to_index("VERB")
      # => 2

      EventPatterns.pattern_tensor("svo_basic")
      # => Nx.tensor([1, 2, 3])  # PRON=1, VERB=2, NOUN=3
  """

  require Logger

  @event_patterns_path "priv/analysis/event_patterns.json"
  @external_resource @event_patterns_path

  @data (case File.read(@event_patterns_path) do
           {:ok, content} ->
             case Jason.decode(content) do
               {:ok, data} -> data
               {:error, reason} ->
                 IO.warn("Failed to parse event_patterns.json: #{inspect(reason)}")
                 %{}
             end

           {:error, _} ->
             IO.warn("event_patterns.json not found, using defaults")
             %{}
         end)

  @patterns Map.get(@data, "patterns", [])
  @pos_tag_indices Map.get(@data, "pos_tag_indices", %{})
  @actor_pos_tags Map.get(@data, "actor_pos_tags", ["PRON", "NOUN", "PROPN"])
  @action_pos_tags Map.get(@data, "action_pos_tags", ["VERB"])
  @object_pos_tags Map.get(@data, "object_pos_tags", ["NOUN", "PROPN", "NUM"])
  @modifier_pos_tags Map.get(@data, "modifier_pos_tags", ["ADV", "ADJ", "ADP"])
  @tense_indicators Map.get(@data, "tense_indicators", %{})

  # Pre-compute pattern tensors at compile time
  @pattern_tensors (
    for pattern <- @patterns, into: %{} do
      structure = Map.get(pattern, "structure", [])
      indices = Enum.map(structure, fn tag -> Map.get(@pos_tag_indices, tag, 0) end)
      {Map.get(pattern, "id"), indices}
    end
  )

  # Pre-compute index sets for fast lookups
  @actor_indices MapSet.new(Enum.map(@actor_pos_tags, fn tag -> Map.get(@pos_tag_indices, tag, 0) end))
  @action_indices MapSet.new(Enum.map(@action_pos_tags, fn tag -> Map.get(@pos_tag_indices, tag, 0) end))
  @object_indices MapSet.new(Enum.map(@object_pos_tags, fn tag -> Map.get(@pos_tag_indices, tag, 0) end))
  @modifier_indices MapSet.new(Enum.map(@modifier_pos_tags, fn tag -> Map.get(@pos_tag_indices, tag, 0) end))

  # ============================================================================
  # Pattern Access
  # ============================================================================

  @doc """
  Returns all loaded event patterns.
  """
  def patterns, do: @patterns

  @doc """
  Returns the number of loaded patterns.
  """
  def pattern_count, do: length(@patterns)

  @doc """
  Gets a pattern by ID.
  """
  def get_pattern(id) when is_binary(id) do
    Enum.find(@patterns, fn p -> Map.get(p, "id") == id end)
  end

  @doc """
  Returns patterns sorted by weight (highest first).
  """
  def patterns_by_weight do
    Enum.sort_by(@patterns, fn p -> Map.get(p, "weight", 0.0) end, :desc)
  end

  # ============================================================================
  # POS Tag Index Mapping
  # ============================================================================

  @doc """
  Converts a POS tag string to its numeric index for tensor operations.

  ## Examples

      iex> EventPatterns.pos_to_index("VERB")
      2

      iex> EventPatterns.pos_to_index("UNKNOWN")
      0
  """
  def pos_to_index(tag) when is_binary(tag) do
    Map.get(@pos_tag_indices, tag, 0)
  end

  @doc """
  Converts a numeric index back to a POS tag string.
  """
  def index_to_pos(index) when is_integer(index) do
    @pos_tag_indices
    |> Enum.find(fn {_tag, idx} -> idx == index end)
    |> case do
      {tag, _} -> tag
      nil -> "UNKNOWN"
    end
  end

  @doc """
  Returns the complete POS tag to index mapping.
  """
  def pos_tag_indices, do: @pos_tag_indices

  @doc """
  Converts a list of POS tags to a list of indices.

  ## Examples

      iex> EventPatterns.pos_tags_to_indices(["PRON", "VERB", "NOUN"])
      [1, 2, 3]
  """
  def pos_tags_to_indices(tags) when is_list(tags) do
    Enum.map(tags, &pos_to_index/1)
  end

  @doc """
  Converts a list of {token, tag} tuples to a list of indices.

  ## Examples

      iex> EventPatterns.tagged_tokens_to_indices([{"I", "PRON"}, {"want", "VERB"}])
      [1, 2]
  """
  def tagged_tokens_to_indices(tagged_tokens) when is_list(tagged_tokens) do
    Enum.map(tagged_tokens, fn
      {_token, tag} -> pos_to_index(tag)
      tag when is_binary(tag) -> pos_to_index(tag)
    end)
  end

  # ============================================================================
  # Pattern Tensor Access
  # ============================================================================

  @doc """
  Returns the pre-computed index list for a pattern (for tensor creation).

  ## Examples

      iex> EventPatterns.pattern_indices("svo_basic")
      [1, 2, 3]  # PRON, VERB, NOUN
  """
  def pattern_indices(pattern_id) when is_binary(pattern_id) do
    Map.get(@pattern_tensors, pattern_id, [])
  end

  @doc """
  Creates an Nx tensor from the pattern structure.

  ## Examples

      iex> EventPatterns.pattern_tensor("svo_basic")
      #Nx.Tensor<s64[3] [1, 2, 3]>
  """
  def pattern_tensor(pattern_id) when is_binary(pattern_id) do
    indices = pattern_indices(pattern_id)
    Nx.tensor(indices, type: :s32)
  end

  @doc """
  Returns all pattern tensors as a map of {pattern_id => Nx.Tensor}.
  """
  def all_pattern_tensors do
    for {id, indices} <- @pattern_tensors, into: %{} do
      {id, Nx.tensor(indices, type: :s32)}
    end
  end

  # ============================================================================
  # Role Detection
  # ============================================================================

  @doc """
  Checks if a POS index represents an actor role.
  """
  def is_actor_index?(index) when is_integer(index) do
    MapSet.member?(@actor_indices, index)
  end

  @doc """
  Checks if a POS index represents an action (verb) role.
  """
  def is_action_index?(index) when is_integer(index) do
    MapSet.member?(@action_indices, index)
  end

  @doc """
  Checks if a POS index represents an object role.
  """
  def is_object_index?(index) when is_integer(index) do
    MapSet.member?(@object_indices, index)
  end

  @doc """
  Checks if a POS index represents a modifier role.
  """
  def is_modifier_index?(index) when is_integer(index) do
    MapSet.member?(@modifier_indices, index)
  end

  @doc """
  Returns the set of actor POS tag indices.
  """
  def actor_indices, do: @actor_indices

  @doc """
  Returns the set of action POS tag indices.
  """
  def action_indices, do: @action_indices

  @doc """
  Returns the set of object POS tag indices.
  """
  def object_indices, do: @object_indices

  @doc """
  Returns the set of modifier POS tag indices.
  """
  def modifier_indices, do: @modifier_indices

  # ============================================================================
  # Tense Detection
  # ============================================================================

  @doc """
  Returns tense indicators configuration.
  """
  def tense_indicators, do: @tense_indicators

  @doc """
  Gets the tense hint for a pattern (if any).
  """
  def pattern_tense_hint(pattern_id) when is_binary(pattern_id) do
    case get_pattern(pattern_id) do
      nil -> nil
      pattern -> Map.get(pattern, "tense_hint")
    end
  end

  # ============================================================================
  # POS Tag Lists
  # ============================================================================

  @doc "Returns the list of POS tags that can be actors."
  def actor_pos_tags, do: @actor_pos_tags

  @doc "Returns the list of POS tags that represent actions."
  def action_pos_tags, do: @action_pos_tags

  @doc "Returns the list of POS tags that can be objects."
  def object_pos_tags, do: @object_pos_tags

  @doc "Returns the list of POS tags that are modifiers."
  def modifier_pos_tags, do: @modifier_pos_tags
end
