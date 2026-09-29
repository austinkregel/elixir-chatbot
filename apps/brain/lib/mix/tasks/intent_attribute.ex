defmodule Mix.Tasks.Intent.Attribute do
  @shortdoc "Attribute the gap between classifier accuracy and pipeline accuracy, per override path"

  @moduledoc """
  Measures where a correct intent prediction stops being the pipeline's answer.

      mix intent.attribute            # report
      mix intent.attribute --verbose  # list every destroyed and rescued row

  `mix evaluate.intent` reports what the pipeline answered. `mix axes.experiment`
  reports what the classifier predicted. The two disagree by about ten points, and
  neither task says which of the pipeline's override paths spends them. This one
  measures both on the same pass over the same rows, so the difference is a
  decomposition rather than a subtraction of two separately-measured figures.

  ## What it records per held-out row

    * `gold` — the label the split says is right
    * `classifier` — top-1 from the deployed `intent_full.term`, scored on the
      vector the pipeline itself built for that row, with its confidence
    * `determined` — the intent and method `determine_intent/6` returned, taken
      from the `:intent_determined` / `:intent_refined` progress events rather than
      reconstructed from the thresholds, so it reports what the code did
    * `final` — `analysis.intent`, after `ContextualEntityInferrer.infer/5`

  `determined` and `final` are recorded separately because
  `ContextualEntityInferrer` can rewrite the intent after `determine_intent/6` has
  returned. Attributing from the thresholds alone would charge its mistakes to the
  domain-conflict branch.

  ## What it does not do

  It changes no threshold and tunes nothing. It reports the confidence
  distribution of correct predictions, since a threshold is only worth moving when
  correct predictions reach the branch it guards.

  Thresholds are read from `Brain.Analysis.Pipeline`, not copied, so the buckets
  follow the deployed values.
  """

  use Mix.Task

  alias Brain.Analysis.{FeatureExtractor, Pipeline}
  alias Brain.ML.{EvaluationStore, FeatureVectorClassifier}

  @requirements ["app.start"]

  @switches [verbose: :boolean]

  @topic "brain:analysis"

  # Below System.schedulers_online() for the same reason mix axes.experiment is:
  # analyze_chunk/2 reaches Brain.Memory.Store.query_semantic through
  # ContextAccumulator, and that is one GenServer with a 5s call timeout.
  @concurrency 4

  # Progress events are broadcast fire-and-forget, so they can still be in flight
  # when the last row's analyze_chunk/2 has already returned.
  @settle_ms 2_000

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("intent.attribute: unknown options #{inspect(invalid)}")
    if positional != [], do: Mix.raise("intent.attribute: unexpected arguments #{inspect(positional)}")

    Mix.shell().info("\nAwaiting MicroClassifiers readiness...")
    Brain.ML.MicroClassifiers.await_ready(:infinity)

    model = load_model!()
    held_out = load_held_out!()

    Phoenix.PubSub.subscribe(Brain.PubSub, @topic)

    rows = measure(held_out, model)
    events = drain_events()

    rows = attach_methods!(rows, events)

    report(rows, opts[:verbose] || false)
  end

  # -- inputs -----------------------------------------------------------------

  defp load_model!() do
    path = Brain.priv_path("ml_models/micro/intent_full.term")

    unless File.regular?(path) do
      Mix.raise("intent.attribute: no intent_full model at #{path}. Run `mix train_micro` first.")
    end

    path |> File.read!() |> :erlang.binary_to_term()
  end

  defp load_held_out!() do
    case EvaluationStore.load_gold_standard("intent", :held_out) do
      [] ->
        Mix.raise("""
        intent.attribute: the held-out split is empty.

        An empty set cannot measure anything, which is not the same as measuring
        zero. Re-carve it with `mix split.held_out intent`.
        """)

      rows ->
        Mix.shell().info("  held-out rows: #{length(rows)}\n")
        rows
    end
  end

  # -- measuring ---------------------------------------------------------------

  defp measure(held_out, model) do
    total = length(held_out)

    rows =
      held_out
      |> Enum.with_index()
      |> Task.async_stream(
        fn {entry, idx} ->
          measure_row(entry, idx, model, total)
        end,
        max_concurrency: @concurrency,
        timeout: 120_000,
        ordered: false
      )
      |> Enum.map(fn {:ok, row} -> row end)

    # A row that failed would quietly make the gap look smaller on both sides.
    if length(rows) != total do
      Mix.raise("intent.attribute: #{total - length(rows)} of #{total} rows failed the pipeline")
    end

    rows
  end

  defp measure_row(entry, idx, model, total) do
    text = entry["text"]
    tag = "intent-attribute-#{idx}"

    if rem(idx + 1, 100) == 0, do: Mix.shell().info("  [#{idx + 1}/#{total}]")

    analysis =
      Pipeline.analyze_chunk(text,
        side_effects: false,
        progress: %{conversation_id: tag, message_id: tag}
      )

    {vector, _word_feats} = FeatureExtractor.extract(analysis)

    {classifier, confidence} =
      case FeatureVectorClassifier.classify(vector, model) do
        {:ok, label, conf, _details} -> {label, conf}
        :error -> {nil, nil}
      end

    %{
      tag: tag,
      provenance: entry["labeled_by"] || "unrecorded",
      text: text,
      gold: entry["intent"],
      classifier: classifier,
      confidence: confidence,
      final: to_string(analysis.intent || "unknown"),
      # The :intent_domain micro-classifier's answer. Pass 1 puts it on the struct
      # and pass 2 carries it through, and it is the same signal ChunkProfile.domain
      # carries into the domain-conflict test.
      predicted_domain: analysis |> Map.get(:intent_domain) |> normalize_domain(),
      pass: Map.get(analysis, :pass)
    }
  end

  defp normalize_domain(nil), do: nil
  defp normalize_domain(:unknown), do: nil
  defp normalize_domain(d) when is_atom(d), do: Atom.to_string(d)
  defp normalize_domain(d) when is_binary(d), do: d

  # -- progress events ---------------------------------------------------------

  # Keyed by message_id. Pass 2 reports :intent_refined after :intent_determined
  # for the same row, and the later event is the one determine_intent/6 last
  # returned, so it wins.
  defp drain_events(acc \\ %{}) do
    receive do
      {:analysis_progress, %{step: step, message_id: id} = payload}
      when step in [:intent_determined, :intent_refined] ->
        keep? = step == :intent_refined or not match?(%{step: :intent_refined}, Map.get(acc, id))
        drain_events(if keep?, do: Map.put(acc, id, payload), else: acc)

      {:analysis_progress, _other} ->
        drain_events(acc)
    after
      @settle_ms -> acc
    end
  end

  defp attach_methods!(rows, events) do
    missing = for row <- rows, not Map.has_key?(events, row.tag), do: row.tag

    if missing != [] do
      Mix.raise("""
      intent.attribute: #{length(missing)} of #{length(rows)} rows produced no intent
      progress event, so their override path is unknown.

      Brain.Analysis.Progress.report/3 rescues everything and returns :ok, so a
      broadcast failure is silent. Without the event the method would have to be
      reconstructed from the thresholds, which cannot separate the domain-conflict
      branch from a later ContextualEntityInferrer rewrite. Refusing to report a
      decomposition that is partly guesswork.
      """)
    end

    Enum.map(rows, fn row ->
      event = Map.fetch!(events, row.tag)

      method = Map.get(event, :intent_method)
      overridden = Map.get(event, :original_intent)
      overridden_score = Map.get(event, :original_score)

      Map.merge(row, %{
        determined: to_string(Map.get(event, :intent) || "unknown"),
        method: method,
        overridden: overridden && to_string(overridden),
        overridden_score: overridden_score,
        branch: branch(method, overridden, overridden_score),
        event_pass: Map.get(event, :pass)
      })
    end)
  end

  # `determine_intent/6` returns :speech_act_fallback from three different
  # branches, so the method alone does not say which decision was taken. The label
  # it overrode and that label's score separate them, using the same two tests the
  # function itself uses.
  defp branch(:speech_act_fallback, nil, _score), do: :speech_act_no_classifier

  defp branch(:speech_act_fallback, overridden, score) do
    cond do
      not Pipeline.registered_intent?(overridden) -> :unregistered_label
      is_number(score) and score >= Pipeline.disambiguation_threshold() -> :domain_conflict
      is_number(score) and score < Pipeline.low_confidence_floor() -> :low_confidence_floor
      # Registered, between the floor and the threshold: the domain-conflict branch
      # is the only one of the three that can be reached there, because the
      # low-confidence clause is guarded by `< @low_confidence_floor`.
      true -> :domain_conflict
    end
  end

  defp branch(method, _overridden, _score), do: method

  # -- reporting ---------------------------------------------------------------

  defp report(rows, verbose?) do
    total = length(rows)
    clf_right = Enum.count(rows, &(&1.classifier == &1.gold))
    det_right = Enum.count(rows, &(&1.determined == &1.gold))
    final_right = Enum.count(rows, &(&1.final == &1.gold))

    line()
    Mix.shell().info("INTENT ATTRIBUTION — #{total} held-out rows")
    line()
    Mix.shell().info("  classifier top-1 correct        #{fmt(clf_right, total)}")
    Mix.shell().info("  after determine_intent/6        #{fmt(det_right, total)}")
    Mix.shell().info("  after entity inference (final)  #{fmt(final_right, total)}")
    Mix.shell().info("")

    report_stage(rows, "determine_intent/6", & &1.classifier, & &1.determined)
    report_stage(rows, "ContextualEntityInferrer", & &1.determined, & &1.final)
    report_divergence(rows)
    report_domain_premise(rows)
    report_calibration(rows)
    report_provenance(rows)
    report_exposure(rows)

    if verbose?, do: report_rows(rows)
  end

  defp report_stage(rows, name, before_fn, after_fn) do
    destroyed = Enum.filter(rows, &(before_fn.(&1) == &1.gold and after_fn.(&1) != &1.gold))
    rescued = Enum.filter(rows, &(before_fn.(&1) != &1.gold and after_fn.(&1) == &1.gold))

    Mix.shell().info("  #{name}: destroyed #{length(destroyed)}, rescued #{length(rescued)}, net #{net(length(rescued) - length(destroyed))}")

    by_method =
      (destroyed ++ rescued)
      |> Enum.group_by(& &1.branch)
      |> Enum.map(fn {method, group} ->
        d = Enum.count(group, &(before_fn.(&1) == &1.gold))
        {method, d, length(group) - d}
      end)
      |> Enum.sort_by(fn {_m, d, _r} -> -d end)

    Enum.each(by_method, fn {method, d, r} ->
      Mix.shell().info("      #{String.pad_trailing(inspect(method), 24)} destroyed #{String.pad_leading(to_string(d), 4)}   rescued #{String.pad_leading(to_string(r), 4)}")
    end)

    Mix.shell().info("")
  end

  # This task scores the deployed model on the vector directly. The pipeline does
  # not use that label as-is: `SpeechActClassifier.refine_with_intent/2` and
  # `apply_profile_rerank/2` both run on the lattice before `determine_intent/6`
  # reads a label off it. Where the method is :classifier, `determined` *is* the
  # label the pipeline classified with, so the two disagreeing on those rows
  # measures what those two steps changed — before any override branch is reached.
  defp report_divergence(rows) do
    kept = Enum.filter(rows, &(&1.branch == :classifier))
    moved = Enum.filter(kept, &(&1.classifier != &1.determined))
    cost = Enum.count(moved, &(&1.classifier == &1.gold and &1.determined != &1.gold))
    gain = Enum.count(moved, &(&1.classifier != &1.gold and &1.determined == &1.gold))

    Mix.shell().info("  Lattice rerank, before any override (#{length(kept)} rows kept the classifier's answer):")
    Mix.shell().info("    label differs from the raw model  #{length(moved)}")
    Mix.shell().info("      raw right -> reranked wrong     #{cost}")
    Mix.shell().info("      raw wrong -> reranked right     #{gain}")
    Mix.shell().info("")
  end

  # The domain-conflict branch discards a registered classifier label whenever the
  # :intent_domain micro-classifier disagrees with it, on the stated grounds that
  # the domain model "is far more reliable than the 232-class intent_full centroid"
  # (pipeline.ex, above classifier_domain_conflicts_with_profile?/2). That premise
  # is testable: the gold label's own prefix says what the domain was.
  defp report_domain_premise(rows) do
    scored = Enum.filter(rows, &is_binary(&1.predicted_domain))

    Mix.shell().info("  :intent_domain premise (#{length(scored)} of #{length(rows)} rows had a domain):")

    if scored != [] do
      right = Enum.count(scored, &(&1.predicted_domain == domain_of(&1.gold)))
      Mix.shell().info("    domain correct, all rows            #{fmt(right, length(scored))}")

      overridden = Enum.filter(scored, &(&1.branch == :domain_conflict))

      if overridden != [] do
        o_right = Enum.count(overridden, &(&1.predicted_domain == domain_of(&1.gold)))

        Mix.shell().info("    domain correct, where it overrode   #{fmt(o_right, length(overridden))}")
      end

      landed = Enum.filter(scored, &(&1.classifier == &1.gold and &1.determined != &1.gold))

      if landed != [] do
        same = Enum.count(landed, &(domain_of(&1.determined) == domain_of(&1.gold)))

        Mix.shell().info("    replacement kept gold's domain      #{fmt(same, length(landed))}  (destroyed rows)")
      end
    end

    Mix.shell().info("")
  end

  defp domain_of(label) when is_binary(label), do: label |> String.split(".") |> List.first()
  defp domain_of(_), do: nil

  # Does the classifier's confidence mean anything? For each decile of reported
  # confidence, the share of those rows it actually got right. A hedge is only
  # worth conditioning on confidence if accuracy tracks it.
  #
  # `gap` is mean confidence minus accuracy: positive is overconfidence. ECE is
  # the row-weighted mean of |gap|.
  defp report_calibration(rows) do
    scored = Enum.filter(rows, &is_number(&1.confidence))

    Mix.shell().info("  Calibration of classifier confidence (#{length(scored)} rows):")

    if scored == [] do
      Mix.shell().info("    no row reported a confidence")
    else
      buckets =
        scored
        |> Enum.group_by(fn row -> min(trunc(row.confidence * 10), 9) end)
        |> Enum.sort_by(&elem(&1, 0))

      Mix.shell().info("    band        rows   mean conf   clf acc   pipeline acc   gap")

      ece =
        Enum.reduce(buckets, 0.0, fn {decile, group}, acc ->
          n = length(group)
          mean_conf = Enum.sum(Enum.map(group, & &1.confidence)) / n
          clf_acc = Enum.count(group, &(&1.classifier == &1.gold)) / n
          pipe_acc = Enum.count(group, &(&1.final == &1.gold)) / n
          gap = mean_conf - clf_acc

          Mix.shell().info(
            "    #{String.pad_trailing("#{decile / 10}-#{(decile + 1) / 10}", 11)} " <>
              "#{String.pad_leading(to_string(n), 5)}   " <>
              "#{String.pad_leading(pct(mean_conf), 8)}   " <>
              "#{String.pad_leading(pct(clf_acc), 7)}   " <>
              "#{String.pad_leading(pct(pipe_acc), 12)}   " <>
              "#{String.pad_leading(pct(gap), 6)}"
          )

          acc + n / length(scored) * abs(gap)
        end)

      Mix.shell().info("")
      Mix.shell().info("    expected calibration error  #{pct(ece)}")

      monotone =
        buckets
        |> Enum.map(fn {_d, g} -> Enum.count(g, &(&1.classifier == &1.gold)) / length(g) end)
        |> then(fn accs -> accs == Enum.sort(accs) end)

      Mix.shell().info("    accuracy rises with confidence across every band: #{monotone}")
    end

    Mix.shell().info("")
  end

  # Accuracy split by where each held-out row came from. The corpus mixes genuine
  # Dialogflow utterances with texts materialized back into the export and with
  # synthetic mutations; scoring them together reports one number for three
  # different things.
  defp report_provenance(rows) do
    groups = Enum.group_by(rows, & &1.provenance)

    Mix.shell().info("  Held-out accuracy by provenance:")
    Mix.shell().info("    origin          rows   clf acc   pipeline acc   mean conf")

    groups
    |> Enum.sort_by(fn {_k, g} -> -length(g) end)
    |> Enum.each(fn {origin, group} ->
      n = length(group)
      clf = Enum.count(group, &(&1.classifier == &1.gold)) / n
      pipe = Enum.count(group, &(&1.final == &1.gold)) / n
      scored = Enum.filter(group, &is_number(&1.confidence))

      conf =
        if scored == [], do: 0.0, else: Enum.sum(Enum.map(scored, & &1.confidence)) / length(scored)

      Mix.shell().info(
        "    #{String.pad_trailing(origin, 14)} #{String.pad_leading(to_string(n), 5)}   " <>
          "#{String.pad_leading(pct(clf), 7)}   #{String.pad_leading(pct(pipe), 12)}   " <>
          "#{String.pad_leading(pct(conf), 9)}"
      )
    end)

    genuine = Map.get(groups, "dialogflow", [])

    if genuine != [] and length(genuine) < length(rows) do
      g_pipe = Enum.count(genuine, &(&1.final == &1.gold)) / length(genuine)
      all_pipe = Enum.count(rows, &(&1.final == &1.gold)) / length(rows)

      Mix.shell().info("")

      Mix.shell().info(
        "    pipeline accuracy on Dialogflow rows only: #{pct(g_pipe)} " <>
          "against #{pct(all_pipe)} reported over all #{length(rows)}"
      )
    end

    Mix.shell().info("")
  end

  # How many correct predictions reach the branches these thresholds guard. A
  # threshold is only worth moving if correct answers pass through it.
  defp report_exposure(rows) do
    floor_at = Pipeline.low_confidence_floor()
    threshold = Pipeline.disambiguation_threshold()

    correct = Enum.filter(rows, &(&1.classifier == &1.gold))
    scored = Enum.filter(correct, &is_number(&1.confidence))

    Mix.shell().info("  Confidence of CORRECT classifier predictions (#{length(correct)} rows):")

    if length(scored) != length(correct) do
      Mix.shell().info("    #{length(correct) - length(scored)} had no confidence and are excluded from the buckets")
    end

    buckets = [
      {"trusted outright (>= #{threshold})", &(&1 >= threshold)},
      {"atlas disambiguation [#{floor_at}, #{threshold})", &(&1 >= floor_at and &1 < threshold)},
      {"speech-act fallback (< #{floor_at})", &(&1 < floor_at)}
    ]

    Enum.each(buckets, fn {label, pred} ->
      group = Enum.filter(scored, &pred.(&1.confidence))
      kept = Enum.count(group, &(&1.final == &1.gold))

      Mix.shell().info(
        "    #{String.pad_trailing(label, 40)} #{String.pad_leading(to_string(length(group)), 4)} rows, #{kept} still right at the end"
      )
    end)

    Mix.shell().info("")
  end

  defp report_rows(rows) do
    destroyed = Enum.filter(rows, &(&1.classifier == &1.gold and &1.final != &1.gold))

    Mix.shell().info("  DESTROYED — classifier had it right, pipeline did not (#{length(destroyed)}):")
    Mix.shell().info("")

    destroyed
    |> Enum.sort_by(& &1.gold)
    |> Enum.each(fn row ->
      Mix.shell().info("    #{row.gold}")
      Mix.shell().info("      text        #{String.slice(row.text || "", 0, 70)}")
      Mix.shell().info("      confidence  #{fmt_conf(row.confidence)}   branch #{inspect(row.branch)}   pass #{inspect(row.pass)}")
      Mix.shell().info("      overrode    #{row.overridden || "-"} @ #{fmt_conf(row.overridden_score)}")
      Mix.shell().info("      determined  #{row.determined}")
      Mix.shell().info("      final       #{row.final}")
      Mix.shell().info("")
    end)
  end

  defp fmt(n, total), do: "#{String.pad_leading(to_string(n), 4)} / #{total}  (#{pct(n / total)})"
  defp pct(x), do: "#{Float.round(x * 100, 1)}%"
  defp net(n) when n >= 0, do: "+#{n}"
  defp net(n), do: to_string(n)
  defp fmt_conf(nil), do: "-"
  defp fmt_conf(c), do: Float.round(c * 1.0, 3) |> to_string()
  defp line, do: Mix.shell().info(String.duplicate("=", 72))
end
