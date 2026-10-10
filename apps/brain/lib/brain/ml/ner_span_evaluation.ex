defmodule Brain.ML.NerSpanEvaluation do
  @moduledoc """
  Span-level scoring of extracted entities against the NER gold standard.

  Each row's gold spans are placed by `Brain.ML.NerGold.spans!/1`. An entity
  is compared by its extent (`:start_pos`..`:end_pos`) and its type mapped onto
  the gold labels (`Brain.ML.EntityExtractor.load_type_mappings/0`):

  - **exact** -- a gold span has the same extent and the same type;
  - **agree** -- a gold span has the same extent and a type that is the
    entity's or refines it in `Brain.Analysis.TypeHierarchy` (a `number` for a
    gold `final_value`), as `Brain.ML.SpanReliability` counts agreement.

  A span extraction left unresolved (`resolved: false`) is an abstention, not
  a prediction: it costs recall, never precision, and is counted on its own.
  Every other entity is a prediction, including entities a later stage of the
  pipeline added.

  The per-source breakdown attributes each prediction to its `:source` (the
  reading a decided span was built from), and counts abstentions by reason.
  """

  @type row_result :: %{gold: [map()], entities: [map()]}

  @doc """
  Scores `results`, one `%{gold: spans, entities: entities}` per row.

  Options: `:canonical` (runtime type => gold label) and `:hierarchy` (a module
  with `is_a?/2`), both required. Returns `%{exact: prf, agree: prf,
  predictions, gold, abstentions, abstentions_by_reason, by_source}` where a
  `prf` is `%{tp, precision, recall, f1}`.
  """
  @spec score([row_result()], keyword()) :: map()
  def score(results, opts) when is_list(results) do
    canonical = Keyword.fetch!(opts, :canonical)
    hierarchy = Keyword.fetch!(opts, :hierarchy)

    judged =
      Enum.flat_map(results, fn %{gold: gold, entities: entities} ->
        Enum.map(entities, fn
          %{resolved: false} = span ->
            {:abstained, Map.get(span, :unresolved_reason)}

          entity ->
            {:predicted, Map.get(entity, :source), judge(entity, gold, canonical, hierarchy)}
        end)
      end)

    gold_total = results |> Enum.map(&length(&1.gold)) |> Enum.sum()
    predictions = for {:predicted, source, verdict} <- judged, do: {source, verdict}
    abstentions = for {:abstained, reason} <- judged, do: reason

    %{
      gold: gold_total,
      predictions: length(predictions),
      exact: prf(Enum.count(predictions, &(elem(&1, 1) == :exact)), length(predictions), gold_total),
      agree: prf(Enum.count(predictions, &(elem(&1, 1) in [:exact, :refined])), length(predictions), gold_total),
      abstentions: length(abstentions),
      abstentions_by_reason: Enum.frequencies(abstentions),
      by_source:
        predictions
        |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
        |> Map.new(fn {source, verdicts} ->
          {source,
           %{
             predictions: length(verdicts),
             exact: Enum.count(verdicts, &(&1 == :exact)),
             agree: Enum.count(verdicts, &(&1 in [:exact, :refined]))
           }}
        end)
    }
  end

  # :exact, :refined (the gold refines the entity's type), or :miss.
  defp judge(entity, gold, canonical, hierarchy) do
    type = entity |> Map.fetch!(:entity_type) |> type!(entity)
    label = Map.get(canonical, type, type)

    case Enum.find(gold, &(&1.start_pos == entity.start_pos and &1.end_pos == entity.end_pos)) do
      %{type: ^label} -> :exact
      %{type: gold_type} -> if hierarchy.is_a?(gold_type, label), do: :refined, else: :miss
      nil -> :miss
    end
  end

  defp type!(type, _entity) when is_binary(type), do: type
  defp type!(type, _entity) when is_atom(type) and not is_nil(type), do: Atom.to_string(type)

  defp type!(_type, entity) do
    raise ArgumentError, "NerSpanEvaluation: a decided entity has no type: #{inspect(entity, limit: 8)}"
  end

  defp prf(tp, predicted, gold) do
    precision = if predicted > 0, do: tp / predicted, else: 0.0
    recall = if gold > 0, do: tp / gold, else: 0.0
    f1 = if precision + recall > 0, do: 2 * precision * recall / (precision + recall), else: 0.0
    %{tp: tp, precision: precision, recall: recall, f1: f1}
  end
end
