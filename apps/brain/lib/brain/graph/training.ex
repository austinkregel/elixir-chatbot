defmodule Brain.Graph.Training do
  @moduledoc """
  Integrates graph data back into ML training pipelines.

  Closes the loop: conversations produce graph data, and graph data
  feeds back into model training to improve future conversations.

  ## Integration Points

  - `conversation_graph` -> Intent Classifier: Topic transition priors

  The gazetteer is deliberately not fed from the graph: the graph holds the
  system's own unreviewed extractions, and the gazetteer learns only what a
  human reviewer approved (see `Brain.Knowledge.ReviewQueue`). Nor is the POS
  tagger: it is a neural model trained on the UD English Web Treebank
  (`Brain.Training.POS`), with no transition table to blend graph counts into.

  ## Blend Ratios

  Graph-derived weights are blended with existing model weights to prevent
  noisy graph data from overwhelming labeled training data:

  - Intent priors: 85% TF-IDF score + 15% graph prior (default)
  """

  alias Atlas.Graph
  alias Atlas.Graph.EdgeLabels
  require Logger

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
    MATCH (a:Topic)-[r:#{EdgeLabels.topic_transition()}]->(b:Topic)
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
