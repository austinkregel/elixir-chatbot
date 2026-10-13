defmodule Mix.Tasks.Evaluate.Ner do
  alias Brain.Analysis.Pipeline
  alias Brain.ML
  @shortdoc "Evaluate named entity recognition accuracy"
  @moduledoc "Evaluate NER against gold standard data.\n\nUses the full production pipeline for evaluation to measure real-world accuracy,\nincluding type normalization and disambiguation.\n\nTwo scores are reported. The entity-level one matches each gold entity to an\nextracted entity by value. The span-level one (`Brain.ML.NerSpanEvaluation`)\nmatches by extent against the gold spans `Brain.ML.NerGold` places, exact and\nup to hierarchy agreement, counts unresolved spans as abstentions, and breaks\nthe decided entities down by source.\n\n## Usage\n\n    mix evaluate.ner --world <id>              # Run evaluation\n    mix evaluate.ner --world <id> --save       # Save results\n    mix evaluate.ner --world <id> --verbose    # Show per-type details\n\n## Options\n\n    --world <id>   Required. The world the gold standard is analyzed in.\n    --save         Save results\n    --verbose      Show per-type details\n\n## Gold Standard Format\n\nAccepts either `\"entities\"` or `\"expected\"` as the key for entity annotations:\n\n    [\n      {\n        \"text\": \"What's the weather in London?\",\n        \"entities\": [{\"value\": \"London\", \"type\": \"location\"}]\n      }\n    ]\n"

  use Mix.Task

  require Logger

  alias ML.{Evaluation, EvaluationStore, NerGold, NerSpanEvaluation}

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, strict: [save: :boolean, verbose: :boolean, world: :string])

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("mix evaluate.ner needs --world <id>, the world the gold standard is analyzed in")

    Mix.Task.run("app.start")

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    ML.MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.")

    save? = Keyword.get(opts, :save, false)
    verbose? = Keyword.get(opts, :verbose, false)

    # The held-out split only: the entity model trains on the rest. Raises when no
    # split has been carved (`mix split.held_out ner`).
    gold = EvaluationStore.load_gold_standard("ner", :held_out)

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("NER EVALUATION (#{length(gold)} examples)")
    IO.puts("(Using production pipeline with type normalization)")
    IO.puts(String.duplicate("=", 60) <> "\n")

    start_time = System.monotonic_time(:millisecond)
    {predictions, actuals, counts, span_results} = evaluate_all(gold, world_id)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    total_entities = counts.ok + counts.errored
    error_rate = counts.errored / max(total_entities, 1)

    if error_rate > 0.10 do
      Mix.raise("""
      Evaluation aborted: NER error rate #{Float.round(error_rate * 100, 1)}% exceeds 10% threshold.
        ok: #{counts.ok}, errored: #{counts.errored}
      """)
    end

    result = Evaluation.build_result("ner", predictions, actuals)

    IO.puts("Entity-level Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:              #{Float.round(result.macro_f1 * 100, 1)}%")
    IO.puts("Duration:              #{duration_ms}ms")

    if counts.errored > 0 do
      IO.puts("Pipeline errors:       #{counts.errored}/#{total_entities} examples failed")
    end

    IO.puts("")

    span_score = report_spans(span_results)
    result = result |> Map.put(:diagnostics, counts) |> Map.put(:span_level, span_score)

    broadcast_result(result, duration_ms)
    Brain.Telemetry.emit_evaluation_complete("ner", %{
      accuracy: result.accuracy,
      macro_f1: result.macro_f1,
      weighted_f1: result.weighted_f1,
      total_examples: result.total_examples,
      duration_ms: duration_ms
    })

    if verbose? do
      cm = Evaluation.confusion_matrix(predictions, actuals)
      report = Evaluation.classification_report(cm)
      IO.puts(Evaluation.format_report(report))
      IO.puts("")
    end

    if save? do
      {:ok, path} = EvaluationStore.save(result)
      IO.puts("Results saved to: #{path}")
    end

    IO.puts("")
  end

  # Span-level scores against the gold spans, and the same split by source.
  defp report_spans(span_results) do
    score =
      NerSpanEvaluation.score(span_results,
        canonical: ML.EntityExtractor.load_type_mappings(),
        hierarchy: Brain.Analysis.TypeHierarchy
      )

    misaligned =
      for %{text: text, entities: entities} <- span_results,
          entity <- entities,
          String.slice(text, entity.start_pos, entity.end_pos - entity.start_pos + 1) != entity.match,
          do: {text, entity.match}

    IO.puts("Span-level (#{score.gold} gold spans, #{score.predictions} decided, #{score.abstentions} unresolved)")
    IO.puts("  exact span+type  P #{pct(score.exact.precision)}  R #{pct(score.exact.recall)}  F1 #{pct(score.exact.f1)}")
    IO.puts("  agree (refines)  P #{pct(score.agree.precision)}  R #{pct(score.agree.recall)}  F1 #{pct(score.agree.f1)}")
    IO.puts("  unresolved by reason: #{inspect(score.abstentions_by_reason)}")

    IO.puts("  by source (decided, exact, agree):")

    score.by_source
    |> Enum.sort_by(fn {source, _} -> inspect(source) end)
    |> Enum.each(fn {source, s} ->
      IO.puts("    #{String.pad_trailing(inspect(source), 18)} #{s.predictions}  #{s.exact}  #{s.agree}")
    end)

    if misaligned != [] do
      IO.puts("  decided entities whose match is not the text at their span: #{length(misaligned)}, e.g. #{inspect(Enum.take(misaligned, 3))}")
    end

    IO.puts("")
    score
  end

  defp pct(rate), do: "#{Float.round(rate * 100, 1)}%"

  defp evaluate_all(gold, world_id) do
    initial_counts = %{ok: 0, errored: 0}

    Enum.reduce(gold, {[], [], initial_counts, []}, fn example, {preds, acts, counts, spans} ->
      text = example["text"]
      expected_entities = example["entities"] || example["expected"] || []

      # The unresolved spans are abstentions: they predict no type, so they are
      # kept apart from the entities and only counted.
      {extracted, unresolved_spans, counts} =
        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false, world_id: world_id)
          # The pipeline emits system types; map them onto the gold standard's labels.
          extracted = ML.EntityExtractor.normalize_entity_types(analysis.entities)
          {extracted, analysis.unresolved_spans, %{counts | ok: counts.ok + 1}}
        rescue
          e ->
            Logger.warning("NER pipeline crashed on #{inspect(text)}: #{Exception.message(e)}")
            {[], [], %{counts | errored: counts.errored + 1}}
        catch
          :exit, reason ->
            Logger.warning("NER pipeline exited on #{inspect(text)}: #{inspect(reason)}")
            {[], [], %{counts | errored: counts.errored + 1}}
        end

      {matched_preds, matched_acts} =
        Enum.reduce(expected_entities, {[], []}, fn expected, {p_acc, a_acc} ->
          expected_value = expected["value"]
          expected_type = expected["type"]

          match =
            Enum.find(extracted, fn ext ->
              ext_value = Map.get(ext, :value) || Map.get(ext, "value")
              normalize_entity_value(ext_value) == normalize_entity_value(expected_value)
            end)

          predicted_type =
            if match do
              Map.get(match, :entity_type) || Map.get(match, "entity_type") || "none"
            else
              "none"
            end

          {[to_string(predicted_type) | p_acc], [to_string(expected_type) | a_acc]}
        end)

      span_result = %{text: text, gold: NerGold.spans!(example), entities: extracted, unresolved_spans: unresolved_spans}

      {Enum.reverse(matched_preds) ++ preds, Enum.reverse(matched_acts) ++ acts, counts, [span_result | spans]}
    end)
    |> then(fn {p, a, counts, spans} -> {Enum.reverse(p), Enum.reverse(a), counts, Enum.reverse(spans)} end)
  end

  defp normalize_entity_value(nil), do: ""

  defp normalize_entity_value(value) when is_binary(value) do
    value |> String.downcase() |> String.trim()
  end

  defp normalize_entity_value(value), do: to_string(value)

  defp broadcast_result(result, duration_ms) do
    Phoenix.PubSub.broadcast(Brain.PubSub, "evaluation:complete",
      {:evaluation_complete, %{
        task: "ner",
        accuracy: result.accuracy,
        macro_f1: result.macro_f1,
        weighted_f1: result.weighted_f1,
        total_examples: result.total_examples,
        duration_ms: duration_ms,
        timestamp: DateTime.utc_now()
      }})
  rescue
    _ -> :ok
  end
end
