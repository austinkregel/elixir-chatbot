defmodule Brain.Analysis.ConsistencyChecker do
  @moduledoc """
  Cross-signal consistency checker that compares independent classification
  signals to detect potential misclassifications.

  Compares up to four independent signals:
  - Fast path intent (heuristic pattern matching via RacingAnalyzer)
  - Analysis intent (LSTM/arbitrated via SpeechActClassifier)
  - NLP intent (TF-IDF via IntentClassifierSimple)
  - Event domain (inferred from event extraction frames)

  When multiple independent signals disagree with the final chosen intent,
  that's a strong indicator of misclassification. Results are emitted via
  telemetry and PubSub for dashboard visibility and downstream processing.
  """

  require Logger

  @pubsub_topic "brain:consistency"

  @type signal :: %{
          source: atom(),
          intent: String.t() | nil,
          domain: String.t() | nil,
          confidence: float() | nil
        }

  @type check_result :: %{
          consistent: boolean(),
          final_intent: String.t() | nil,
          signals: [signal()],
          agreeing_signals: [atom()],
          dissenting_signals: [atom()],
          consensus_intent: String.t() | nil,
          consensus_domain: String.t() | nil,
          severity: :none | :low | :medium | :high
        }

  @doc """
  Checks consistency across all available classification signals.

  Returns a check result with details about which signals agree/disagree
  and the severity of any inconsistency.
  """
  @spec check(map()) :: check_result()
  def check(params) do
    final_intent = params[:final_intent]
    signals = build_signals(params)
    non_nil_signals = Enum.filter(signals, &(&1.intent != nil))

    if length(non_nil_signals) < 2 do
      %{
        consistent: true,
        final_intent: final_intent,
        signals: signals,
        agreeing_signals: Enum.map(non_nil_signals, & &1.source),
        dissenting_signals: [],
        consensus_intent: final_intent,
        consensus_domain: nil,
        severity: :none
      }
    else
      analyze_consistency(final_intent, non_nil_signals, signals)
    end
  end

  @doc """
  Runs the consistency check and emits telemetry/PubSub events if
  inconsistency is detected. Call this from the pipeline after intent
  determination.
  """
  @spec check_and_report(map(), keyword()) :: check_result()
  def check_and_report(params, opts \\ []) do
    result = check(params)

    if not result.consistent do
      metadata = %{
        text: params[:text],
        final_intent: result.final_intent,
        consensus_intent: result.consensus_intent,
        severity: result.severity,
        agreeing: result.agreeing_signals,
        dissenting: result.dissenting_signals,
        signals: Enum.map(result.signals, fn s -> {s.source, s.intent} end)
      }

      :telemetry.execute(
        [:chat_bot, :analysis, :consistency, :disagreement],
        %{signal_count: length(result.signals), dissent_count: length(result.dissenting_signals)},
        metadata
      )

      if result.severity in [:medium, :high] do
        Logger.warning("Classification inconsistency (#{result.severity}): " <>
          "final=#{result.final_intent}, consensus=#{result.consensus_intent}, " <>
          "dissenting=#{inspect(result.dissenting_signals)}")
      end

      broadcast_disagreement(metadata, opts)
    end

    result
  end

  defp build_signals(params) do
    [
      build_fast_path_signal(params[:fast_path]),
      build_analysis_signal(params[:analysis_intent], params[:analysis_confidence]),
      build_nlp_signal(params[:nlp_intent], params[:nlp_confidence]),
      build_event_signal(params[:events])
    ]
  end

  defp build_fast_path_signal(nil), do: %{source: :fast_path, intent: nil, domain: nil, confidence: nil}
  defp build_fast_path_signal(%{intent: intent} = fp) do
    %{
      source: :fast_path,
      intent: to_str(intent),
      domain: to_str(fp[:domain]),
      confidence: fp[:activation] || fp[:confidence]
    }
  end
  defp build_fast_path_signal(_), do: %{source: :fast_path, intent: nil, domain: nil, confidence: nil}

  defp build_analysis_signal(nil, _), do: %{source: :analysis, intent: nil, domain: nil, confidence: nil}
  defp build_analysis_signal(intent, confidence) do
    %{
      source: :analysis,
      intent: to_str(intent),
      domain: extract_domain(to_str(intent)),
      confidence: confidence
    }
  end

  defp build_nlp_signal(nil, _), do: %{source: :nlp, intent: nil, domain: nil, confidence: nil}
  defp build_nlp_signal(intent, confidence) do
    %{
      source: :nlp,
      intent: to_str(intent),
      domain: extract_domain(to_str(intent)),
      confidence: confidence
    }
  end

  defp build_event_signal(nil), do: %{source: :events, intent: nil, domain: nil, confidence: nil}
  defp build_event_signal([]), do: %{source: :events, intent: nil, domain: nil, confidence: nil}
  defp build_event_signal(events) when is_list(events) do
    primary_event = List.first(events)
    object_text = if primary_event.object, do: primary_event.object.text
    object = to_str(object_text)

    %{
      source: :events,
      intent: nil,
      domain: if(object != "", do: object),
      confidence: primary_event.confidence
    }
  end
  defp build_event_signal(_), do: %{source: :events, intent: nil, domain: nil, confidence: nil}

  defp analyze_consistency(final_intent, non_nil_signals, all_signals) do
    final_domain = extract_domain(to_str(final_intent))

    {agreeing, dissenting} =
      Enum.split_with(non_nil_signals, fn signal ->
        intent_matches?(signal.intent, final_intent) or
          (signal.intent == nil and domain_matches?(signal.domain, final_domain))
      end)

    domain_signals = Enum.filter(all_signals, &(&1.domain != nil))
    {domain_agreeing, domain_dissenting} =
      Enum.split_with(domain_signals, fn signal ->
        domain_matches?(signal.domain, final_domain)
      end)

    consensus = find_consensus(non_nil_signals, domain_signals)
    total_signals = length(non_nil_signals) + length(domain_signals -- non_nil_signals)
    dissent_count = length(dissenting) + length(domain_dissenting -- dissenting)

    severity =
      cond do
        dissent_count == 0 -> :none
        dissent_count == 1 and total_signals <= 2 -> :low
        dissent_count >= 2 and consensus.intent != nil and consensus.intent != to_str(final_intent) -> :high
        dissent_count >= 1 and consensus.domain != nil and consensus.domain != final_domain -> :medium
        true -> :low
      end

    %{
      consistent: severity == :none,
      final_intent: final_intent,
      signals: all_signals,
      agreeing_signals: Enum.map(agreeing ++ domain_agreeing, & &1.source) |> Enum.uniq(),
      dissenting_signals: Enum.map(dissenting ++ domain_dissenting, & &1.source) |> Enum.uniq(),
      consensus_intent: consensus.intent,
      consensus_domain: consensus.domain,
      severity: severity
    }
  end

  defp find_consensus(intent_signals, domain_signals) do
    intent_votes =
      intent_signals
      |> Enum.filter(&(&1.intent != nil))
      |> Enum.frequencies_by(& &1.intent)

    domain_votes =
      (intent_signals ++ domain_signals)
      |> Enum.uniq_by(& &1.source)
      |> Enum.filter(&(&1.domain != nil))
      |> Enum.frequencies_by(& &1.domain)

    consensus_intent =
      case Enum.max_by(intent_votes, fn {_k, v} -> v end, fn -> nil end) do
        {intent, count} when count >= 2 -> intent
        _ -> nil
      end

    consensus_domain =
      case Enum.max_by(domain_votes, fn {_k, v} -> v end, fn -> nil end) do
        {domain, count} when count >= 2 -> domain
        _ -> nil
      end

    %{intent: consensus_intent, domain: consensus_domain}
  end

  defp intent_matches?(nil, _), do: false
  defp intent_matches?(_, nil), do: false
  defp intent_matches?(a, b), do: to_str(a) == to_str(b)

  defp domain_matches?(nil, _), do: false
  defp domain_matches?(_, nil), do: false
  defp domain_matches?(a, b), do: to_str(a) == to_str(b)

  defp extract_domain(nil), do: nil
  defp extract_domain(""), do: nil
  defp extract_domain(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain, _] -> domain
      _ -> intent
    end
  end

  defp to_str(nil), do: ""
  defp to_str(val) when is_atom(val), do: Atom.to_string(val)
  defp to_str(val) when is_binary(val), do: val
  defp to_str(val), do: inspect(val)

  defp broadcast_disagreement(metadata, _opts) do
    Phoenix.PubSub.broadcast(
      Brain.PubSub,
      @pubsub_topic,
      {:consistency_disagreement, metadata}
    )
  rescue
    _ -> :ok
  end
end
