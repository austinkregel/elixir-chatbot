defmodule Brain.Graph.Training do
  @moduledoc """
  Integrates graph data back into ML training pipelines.

  Closes the loop: conversations produce graph data, and graph data
  feeds back into model training to improve future conversations.

  ## Integration Points

  - `pos_graph` -> POS Tagger: Transition weight refresh from accumulated tag patterns
  - `knowledge_graph` -> Gazetteer: Entity sync from discovered entities
  - `conversation_graph` -> Intent Classifier: Topic transition priors

  ## Blend Ratios

  Graph-derived weights are blended with existing model weights to prevent
  noisy graph data from overwhelming labeled training data:

  - POS weights: 70% existing + 30% graph (default)
  - Gazetteer: Additive (graph entities supplement, don't replace)
  - Intent priors: 85% TF-IDF score + 15% graph prior (default)
  """

  alias Atlas.Graph
  require Logger

  # ============================================================================
  # POS Tagger Weight Refresh
  # ============================================================================

  @doc """
  Refresh POS tagger weights from pos_graph data.

  Exports FOLLOWED_BY edge frequencies as transition weights and
  HAS_TAG edge counts as tag priors, then blends with existing model.

  ## Options

  - `:blend` -- blend ratio for graph weights (default: 0.3)
  """
  def refresh_pos_weights(opts \\ []) do
    blend = Keyword.get(opts, :blend, 0.3)

    with {:ok, transitions} <- fetch_pos_transitions(),
         {:ok, tag_counts} <- fetch_tag_counts() do
      transition_weights = build_transition_weights(transitions)
      tag_priors = build_tag_priors(tag_counts)

      Brain.ML.POSTagger.update_weights(transition_weights, tag_priors, blend: blend)
    else
      {:error, reason} ->
        Logger.warning("Failed to refresh POS weights from graph", reason: inspect(reason))
        {:error, reason}
    end
  end

  defp fetch_pos_transitions do
    query = "MATCH (a:POSTag)-[r:FOLLOWED_BY]->(b:POSTag) RETURN a, b, r"

    case Graph.cypher("pos_graph", query) do
      {:ok, rows} ->
        parsed =
          Enum.map(rows, fn
            [%Atlas.Graph.Types.Vertex{properties: a_props}, %Atlas.Graph.Types.Vertex{properties: b_props}, %Atlas.Graph.Types.Edge{properties: r_props}] ->
              [Map.get(a_props, "name", ""), Map.get(b_props, "name", ""), Map.get(r_props, "frequency", 0)]

            _ ->
              nil
          end)
          |> Enum.reject(&is_nil/1)

        {:ok, parsed}

      error ->
        {:error, error}
    end
  rescue
    e -> {:error, e}
  end

  defp fetch_tag_counts do
    query = "MATCH (:Token)-[r:HAS_TAG]->(t:POSTag) RETURN t, r"

    case Graph.cypher("pos_graph", query) do
      {:ok, rows} ->
        parsed =
          Enum.reduce(rows, %{}, fn
            [%Atlas.Graph.Types.Vertex{properties: t_props}, %Atlas.Graph.Types.Edge{properties: r_props}], acc ->
              tag = Map.get(t_props, "name", "")
              count = Map.get(r_props, "count", 0)
              Map.update(acc, tag, count, &(&1 + count))

            _, acc ->
              acc
          end)
          |> Enum.map(fn {tag, count} -> [tag, count] end)

        {:ok, parsed}

      error ->
        {:error, error}
    end
  rescue
    e -> {:error, e}
  end

  defp build_transition_weights(rows) when is_list(rows) do
    Enum.reduce(rows, %{}, fn
      [from_tag, to_tag, freq], acc when is_binary(from_tag) and is_binary(to_tag) ->
        inner = Map.get(acc, from_tag, %{})
        Map.put(acc, from_tag, Map.put(inner, to_tag, freq || 0))

      _, acc ->
        acc
    end)
  end

  defp build_transition_weights(_), do: %{}

  defp build_tag_priors(rows) when is_list(rows) do
    raw =
      Enum.reduce(rows, %{}, fn
        [tag, count], acc when is_binary(tag) -> Map.put(acc, tag, count || 0)
        _, acc -> acc
      end)

    total = Enum.sum(Map.values(raw))

    if total > 0 do
      Map.new(raw, fn {tag, count} -> {tag, count / total} end)
    else
      %{}
    end
  end

  defp build_tag_priors(_), do: %{}

  # ============================================================================
  # Gazetteer Sync
  # ============================================================================

  @doc """
  Collects gazetteer entries from the knowledge_graph without writing them.

  Returns `{:ok, entries}` where entries is a list of `{name, entity_type, metadata}` tuples,
  or `{:error, reason}`. The caller (Gazetteer) is responsible for inserting into ETS,
  avoiding the self-call deadlock that occurs when a GenServer calls its own public API.
  """
  def collect_gazetteer_entries do
    labels = discover_entity_labels()

    if labels == [] do
      Logger.debug("No entity labels found in knowledge_graph, skipping Gazetteer sync")
      {:ok, []}
    else
      entries =
        Enum.flat_map(labels, fn label ->
          entity_type = label_to_entity_type(label)
          collect_label_entries(label, entity_type)
        end)

      Logger.info("Collected #{length(entries)} gazetteer entries from knowledge_graph")
      {:ok, entries}
    end
  rescue
    e ->
      Logger.warning("Gazetteer entry collection failed", reason: inspect(e))
      {:error, e}
  end

  @doc """
  Sync knowledge_graph entities into the Gazetteer ETS table.

  Safe to call from outside the Gazetteer process. Do NOT call from within
  the Gazetteer GenServer -- use `collect_gazetteer_entries/0` instead.
  """
  def sync_gazetteer do
    case collect_gazetteer_entries() do
      {:ok, entries} ->
        Enum.each(entries, fn {name, entity_type, metadata} ->
          Brain.ML.Gazetteer.add_entry(name, entity_type, metadata)
        end)

        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Discovers all distinct node labels present in the knowledge_graph.

  Returns a list of label strings (e.g., ["Location", "Person", "Device"]).
  """
  def discover_entity_labels do
    query = "MATCH (n) RETURN DISTINCT labels(n) AS lbls"

    case Graph.cypher("knowledge_graph", query) do
      {:ok, rows} ->
        rows
        |> Enum.flat_map(fn
          [labels] when is_list(labels) -> labels
          _ -> []
        end)
        |> Enum.uniq()
        |> Enum.reject(&(&1 in ["_internal", ""]))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp collect_label_entries(label, entity_type) do
    query = "MATCH (n:#{label}) RETURN n"

    case Graph.cypher("knowledge_graph", query) do
      {:ok, rows} ->
        Enum.flat_map(rows, fn
          [%Atlas.Graph.Types.Vertex{properties: props}] ->
            name = Map.get(props, "name", "")

            if name != "" do
              metadata = Map.drop(props, ["name"])
              [{name, entity_type, metadata}]
            else
              []
            end

          _ ->
            []
        end)

      _ ->
        []
    end
  end

  defp label_to_entity_type(label) do
    label
    |> String.downcase()
  end

  # ============================================================================
  # Intent Classification Priors
  # ============================================================================

  @doc """
  Extract intent transition priors from conversation_graph.

  Builds a transition probability matrix from TOPIC_TRANSITION edges:
  `%{"weather.query" => %{"weather.followup" => 0.4, "greeting" => 0.1}, ...}`

  These can be used to boost intent classification confidence when the
  previous intent is known.
  """
  def extract_intent_priors do
    query = """
    MATCH (a:Topic)-[r:TOPIC_TRANSITION]->(b:Topic)
    RETURN a, b, r
    """

    case Graph.cypher("conversation_graph", query) do
      {:ok, rows} when is_list(rows) ->
        parsed =
          Enum.map(rows, fn
            [%Atlas.Graph.Types.Vertex{properties: a_props}, %Atlas.Graph.Types.Vertex{properties: b_props}, %Atlas.Graph.Types.Edge{properties: r_props}] ->
              [Map.get(a_props, "name", ""), Map.get(b_props, "name", ""), Map.get(r_props, "count", 1)]

            _ ->
              nil
          end)
          |> Enum.reject(&is_nil/1)

        build_intent_transition_matrix(parsed)

      _ ->
        %{}
    end
  rescue
    _ -> %{}
  end

  @doc """
  Apply intent transition priors to a set of classification scores.

  ## Parameters

  - `scores` -- list of `{intent, score}` tuples
  - `prev_intent` -- the previous intent in the conversation
  - `priors` -- transition matrix from `extract_intent_priors/0`
  - `weight` -- how much to weight the prior (default: 0.15)

  Returns updated scores with prior-boosted values.
  """
  def apply_intent_priors(scores, prev_intent, priors, opts \\ []) do
    weight = Keyword.get(opts, :weight, 0.15)

    case Map.get(priors, to_string(prev_intent)) do
      nil ->
        scores

      transitions ->
        total = Enum.sum(Map.values(transitions))

        if total > 0 do
          Enum.map(scores, fn {intent, score} ->
            prior = Map.get(transitions, to_string(intent), 0) / total
            boosted = score * (1 - weight) + prior * weight
            {intent, boosted}
          end)
        else
          scores
        end
    end
  end

  defp build_intent_transition_matrix(rows) do
    raw =
      Enum.reduce(rows, %{}, fn
        [from, to, count], acc when is_binary(from) and is_binary(to) ->
          inner = Map.get(acc, from, %{})
          Map.put(acc, from, Map.put(inner, to, count || 1))

        _, acc ->
          acc
      end)

    raw
  end
end
