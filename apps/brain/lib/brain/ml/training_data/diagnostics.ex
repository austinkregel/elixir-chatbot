defmodule Brain.ML.TrainingData.Diagnostics do
  @moduledoc """
  Cross-source diagnostics for training data health.

  Reports class skew, orphan intents (present in one source but missing
  from another), and cross-source lookups.
  """

  alias Brain.ML.TrainingData.{Catalog, SourceDescriptors}

  # ── Skew report ──────────────────────────────────────────────────────

  @doc """
  Compute class distribution with skew warnings for a source.

  Returns a list of `%{label, count, pct, skew_warning}` maps sorted by
  count descending. `skew_warning` is non-nil when ratio vs median exceeds
  `threshold` (default 10x).
  """
  @spec skew_report(atom(), float()) :: {:ok, [map()]} | {:error, term()}
  def skew_report(source_id, threshold \\ 10.0) do
    case Catalog.class_distribution(source_id) do
      {:ok, dist} when map_size(dist) > 0 ->
        total = dist |> Map.values() |> Enum.sum()
        counts = Map.values(dist) |> Enum.sort()
        median = Enum.at(counts, div(length(counts), 2)) |> max(1)

        rows =
          dist
          |> Enum.map(fn {label, count} ->
            pct = if total > 0, do: Float.round(count / total * 100, 1), else: 0.0
            ratio = count / median

            skew_warning =
              cond do
                ratio >= threshold -> "#{Float.round(ratio, 1)}x median — overrepresented"
                1 / max(ratio, 0.001) >= threshold -> "#{Float.round(1 / ratio, 1)}x below median — underrepresented"
                true -> nil
              end

            %{label: label, count: count, pct: pct, skew_warning: skew_warning}
          end)
          |> Enum.sort_by(& &1.count, :desc)

        {:ok, rows}

      {:ok, _} ->
        {:ok, []}

      err ->
        err
    end
  end

  # ── Orphan report ────────────────────────────────────────────────────

  @doc """
  Find orphan intents across the gold standard and intent registry.

  Returns a map with:
  - `:gold_without_registry` — intents in gold that have no registry entry
  - `:registry_without_gold` — registry entries with zero gold examples
  - `:registry_without_templates` — registry entries missing from template store
  """
  @spec orphan_report() :: {:ok, map()} | {:error, term()}
  def orphan_report do
    with {:ok, gold_records} <- Catalog.read_source(:intent_gold),
         {:ok, registry_map} <- Catalog.read_source(:intent_registry) do
      gold_intents =
        gold_records
        |> Enum.map(&Map.get(&1, "intent"))
        |> Enum.reject(&is_nil/1)
        |> MapSet.new()

      registry_intents = registry_map |> Map.keys() |> MapSet.new()

      gold_without_registry =
        MapSet.difference(gold_intents, registry_intents) |> MapSet.to_list() |> Enum.sort()

      registry_without_gold =
        MapSet.difference(registry_intents, gold_intents) |> MapSet.to_list() |> Enum.sort()

      {:ok,
       %{
         gold_without_registry: gold_without_registry,
         registry_without_gold: registry_without_gold,
         gold_intent_count: MapSet.size(gold_intents),
         registry_intent_count: MapSet.size(registry_intents)
       }}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  # ── Cross-source lookup ──────────────────────────────────────────────

  @doc """
  Find every source file that contains a given intent/label string.

  Returns `[%{source_id, source_label, count, sample_texts}]`.
  """
  @spec cross_source_lookup(String.t()) :: [map()]
  def cross_source_lookup(query) when is_binary(query) do
    down = String.downcase(query)

    SourceDescriptors.all()
    |> Enum.filter(&(&1.tag != :external_corpus))
    |> Enum.flat_map(fn desc ->
      case Catalog.read_source(desc.id) do
        {:ok, records} when is_list(records) ->
          matches =
            Enum.filter(records, fn rec ->
              matches_label_or_text?(rec, down, desc)
            end)

          if matches == [] do
            []
          else
            sample_texts =
              matches
              |> Enum.take(5)
              |> Enum.map(fn rec ->
                Map.get(rec, "text") || Map.get(rec, "intent") || Map.get(rec, "label") || "?"
              end)

            [
              %{
                source_id: desc.id,
                source_label: desc.label,
                count: length(matches),
                sample_texts: sample_texts
              }
            ]
          end

        {:ok, %{} = map} ->
          if Map.has_key?(map, query) do
            [
              %{
                source_id: desc.id,
                source_label: desc.label,
                count: 1,
                sample_texts: [query]
              }
            ]
          else
            []
          end

        _ ->
          []
      end
    end)
  end

  # ── Summary stats ────────────────────────────────────────────────────

  @doc "Quick summary stats for the studio sidebar."
  @spec summary_stats() :: map()
  def summary_stats do
    sources = Catalog.list_sources()
    total_files = length(sources)
    existing = Enum.count(sources, & &1.exists)
    total_records = Enum.reduce(sources, 0, fn s, acc -> acc + (s.record_count || 0) end)
    editable = Enum.count(sources, fn s -> SourceDescriptors.editable?(s) end)
    read_only = total_files - editable

    %{
      total_files: total_files,
      existing_files: existing,
      missing_files: total_files - existing,
      total_records: total_records,
      editable: editable,
      read_only: read_only
    }
  end

  # ── Trace prediction ──────────────────────────────────────────────────

  @doc """
  Run a full pipeline trace on an utterance and find the K nearest gold
  examples by feature-vector cosine similarity.

  Returns `{:ok, trace_result}` with:
  - `:intent` — chosen intent
  - `:intent_domain` — domain classifier output
  - `:speech_act` — speech act result summary
  - `:confidence` — intent confidence
  - `:strategy` — overall strategy
  - `:classifier_raw` — raw classifier indicator
  - `:entities` — extracted entities
  - `:missing_context` — missing slots
  - `:neighbors` — top-K gold entries by cosine similarity
  """
  @spec trace_prediction(String.t(), non_neg_integer()) :: {:ok, map()} | {:error, term()}
  def trace_prediction(text, top_k \\ 5) when is_binary(text) and text != "" do
    model = Brain.Analysis.Pipeline.process(text)
    [a | _] = model.analyses

    sa = a.speech_act

    classifier_raw =
      Enum.find_value(sa.indicators || [], fn s ->
        case String.split(to_string(s), ":", parts: 2) do
          ["intent", v] -> v
          _ -> nil
        end
      end) || "—"

    fv = a.feature_vector

    neighbors =
      if is_list(fv) and fv != [] do
        find_nearest_gold(fv, top_k)
      else
        []
      end

    {:ok,
     %{
       intent: a.intent,
       intent_domain: a.intent_domain,
       speech_act_category: sa.category,
       speech_act_sub_type: sa.sub_type,
       is_question: sa.is_question,
       confidence: sa.intent_confidence,
       strategy: model.overall_strategy,
       classifier_raw: classifier_raw,
       entities: Enum.map(a.entities, fn e -> %{entity: e[:entity], type: e[:entity_type], value: e[:value]} end),
       missing_context: a.missing_context,
       neighbors: neighbors
     }}
  rescue
    e ->
      {:error, Exception.message(e)}
  end

  defp find_nearest_gold(query_fv, top_k) do
    case Catalog.read_source(:intent_gold) do
      {:ok, records} when is_list(records) ->
        records
        |> Enum.flat_map(fn rec ->
          case Map.get(rec, "feature_vector") do
            fv when is_list(fv) and fv != [] ->
              sim = FourthWall.Math.cosine_similarity(query_fv, fv)
              [%{text: Map.get(rec, "text", ""), intent: Map.get(rec, "intent", ""), similarity: Float.round(sim, 4)}]

            _ ->
              []
          end
        end)
        |> Enum.sort_by(& &1.similarity, :desc)
        |> Enum.take(top_k)

      _ ->
        []
    end
  end

  # ── Private ─────────────────────────────────────────────────────────

  defp matches_label_or_text?(rec, query, %{record_kind: :intent_example}) do
    text = Map.get(rec, "text", "")
    intent = Map.get(rec, "intent", "") || Map.get(rec, "speech_act", "") || Map.get(rec, "sentiment", "")
    String.contains?(String.downcase(text), query) or String.downcase(to_string(intent)) == query
  end

  defp matches_label_or_text?(rec, query, %{record_kind: kind})
       when kind in [:text_classifier_row, :fv_classifier_row] do
    text = Map.get(rec, "text", "")
    label = Map.get(rec, "label", "")
    String.contains?(String.downcase(text), query) or String.downcase(label) == query
  end

  defp matches_label_or_text?(rec, query, _) do
    rec
    |> Map.values()
    |> Enum.any?(fn
      v when is_binary(v) -> String.contains?(String.downcase(v), query)
      _ -> false
    end)
  end
end
