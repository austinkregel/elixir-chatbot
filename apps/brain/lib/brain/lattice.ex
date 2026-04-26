defmodule Brain.Lattice do
  @moduledoc """
  Multi-hypothesis / N-best wire format: ranked `Brain.Lattice.Candidate` values
  with margin, entropy, and combinators for reranking and fusion.

  Used by intent classification (`:intent_full`) and autonomous learning
  (competing `Brain.Knowledge.Types.Hypothesis` values).
  """

  alias Brain.Lattice.Candidate

  defstruct candidates: [],
            best: nil,
            second: nil,
            margin: 0.0,
            entropy: 0.0,
            source: nil,
            stage: nil,
            error: nil

  @type t :: %__MODULE__{
          candidates: [Candidate.t()],
          best: Candidate.t() | nil,
          second: Candidate.t() | nil,
          margin: float(),
          entropy: float(),
          source: atom() | nil,
          stage: atom() | nil,
          error: term() | nil
        }

  @merge_modes [:weighted_sum, :log_odds, :max]

  @doc "Empty lattice (no candidates). Use `opts[:error]` for tombstone metadata."
  def empty(stage \\ nil, opts \\ []) do
    %__MODULE__{
      candidates: [],
      best: nil,
      second: nil,
      margin: 0.0,
      entropy: 0.0,
      stage: stage,
      error: Keyword.get(opts, :error)
    }
  end

  def empty?(%__MODULE__{candidates: []}), do: true
  def empty?(%__MODULE__{}), do: false

  def error(%__MODULE__{error: e}), do: e

  def best(%__MODULE__{best: b}), do: b

  def second(%__MODULE__{second: s}), do: s

  def margin(%__MODULE__{margin: m}), do: m

  def entropy(%__MODULE__{entropy: e}), do: e

  def best_label(%__MODULE__{best: nil}), do: nil
  def best_label(%__MODULE__{best: %Candidate{label: l}}), do: l

  def to_top_k(%__MODULE__{candidates: cands}) do
    Enum.map(cands, fn %Candidate{label: label, confidence: conf} ->
      intent = if is_binary(label), do: label, else: to_string(label)
      %{intent: intent, score: conf}
    end)
  end

  def to_map(%__MODULE__{} = l) do
    top_3 =
      l.candidates
      |> Enum.take(3)
      |> Enum.map(fn %Candidate{label: lab, confidence: c, score: s} ->
        %{label: lab, confidence: c, score: s}
      end)

    %{
      stage: l.stage,
      error: l.error,
      margin: l.margin,
      entropy: l.entropy,
      candidate_count: length(l.candidates),
      best_label: best_label(l),
      best_confidence: if(l.best, do: l.best.confidence, else: nil),
      top_3: top_3
    }
  end

  def to_context_signal(%__MODULE__{} = l, source) when is_atom(source) do
    case best(l) do
      nil -> {source, :unknown, 0.5}
      %Candidate{label: v, confidence: c} -> {source, v, c}
    end
  end

  @doc """
  Build from classifier result `{:ok, label, confidence, details}`.

  Uses `details[:top_k]` as `{label, score}` pairs; winner gets API `confidence`,
  others get softmax shares of the remaining mass derived from `score`.
  """
  def from_classifier({:ok, label, confidence, details}, opts \\ []) when is_map(details) do
    stage = Keyword.get(opts, :stage)
    source = Keyword.get(opts, :source, :classifier)
    top_k = Map.get(details, :top_k, [])

    candidates =
      case top_k do
        [] ->
          [
            %Candidate{
              label: label,
              score: Map.get(details, :top_score, confidence),
              confidence: clamp01(confidence),
              source: source,
              metadata: %{}
            }
          ]

        pairs ->
          base =
            Enum.flat_map(pairs, fn
              {l, s} when is_binary(l) or is_atom(l) ->
                [%Candidate{label: l, score: s * 1.0, confidence: 0.0, source: source, metadata: %{}}]

              %{label: l, score: s} ->
                [%Candidate{label: l, score: s * 1.0, confidence: 0.0, source: source, metadata: %{}}]

              _ ->
                []
            end)

          scores = Enum.map(base, & &1.score)
          shares = softmax(scores)
          c_w = clamp01(confidence)

          loser_soft_sum =
            base
            |> Enum.zip(shares)
            |> Enum.filter(fn {c, _} -> not labels_equal?(c.label, label) end)
            |> Enum.map(&elem(&1, 1))
            |> Enum.sum()

          Enum.zip(base, shares)
          |> Enum.map(fn {c, p} ->
            conf =
              if labels_equal?(c.label, label) do
                c_w
              else
                rem = max(0.0, 1.0 - c_w)

                if loser_soft_sum > 0.0 do
                  clamp01(rem * p / loser_soft_sum)
                else
                  clamp01(rem / max(1, length(base) - 1))
                end
              end

            %{c | confidence: conf}
          end)
      end

    lattice =
      %__MODULE__{
        candidates: candidates,
        source: source,
        stage: stage,
        error: nil
      }
      |> finalize()

    emit_telemetry(lattice)
    lattice
  end

  defp labels_equal?(a, b) do
    to_string(a) == to_string(b)
  end

  defp emit_telemetry(%__MODULE__{} = l) do
    top_3 =
      l.candidates
      |> Enum.take(3)
      |> Enum.map(fn %Candidate{label: lab, confidence: c} ->
        %{label: lab, confidence: c}
      end)

    :telemetry.execute(
      [:chat_bot, :brain, :lattice, :resolved],
      %{count: 1},
      %{
        stage: l.stage,
        best_label: best_label(l),
        best_confidence: if(l.best, do: l.best.confidence, else: nil),
        margin: l.margin,
        entropy: l.entropy,
        top_3: top_3,
        timestamp: System.monotonic_time(:millisecond)
      }
    )
  end

  def from_top_k(top_k, opts \\ []) when is_list(top_k) do
    source = Keyword.get(opts, :source)
    stage = Keyword.get(opts, :stage)

    cands =
      Enum.flat_map(top_k, fn
        {l, s} ->
          [%Candidate{label: l, score: s * 1.0, confidence: 0.0, source: source, metadata: %{}}]

        %{label: l, score: s} ->
          [%Candidate{label: l, score: s * 1.0, confidence: 0.0, source: source, metadata: %{}}]

        _ ->
          []
      end)

    filled =
      if cands == [] do
        []
      else
        scores = Enum.map(cands, & &1.score)
        soft = softmax(scores)

        Enum.zip(cands, soft)
        |> Enum.map(fn {c, p} -> %{c | confidence: clamp01(p)} end)
      end

    lattice = %__MODULE__{candidates: filled, source: source, stage: stage, error: nil} |> finalize()
    emit_telemetry(lattice)
    lattice
  end

  def singleton(label, score, opts \\ []) do
    conf = Keyword.get(opts, :confidence, infer_confidence_from_score(score))
    source = Keyword.get(opts, :source)
    stage = Keyword.get(opts, :stage)

    lattice =
      %__MODULE__{
        candidates: [
          %Candidate{label: label, score: score * 1.0, confidence: clamp01(conf), source: source, metadata: %{}}
        ],
        source: source,
        stage: stage,
        error: nil
      }
      |> finalize()

    emit_telemetry(lattice)
    lattice
  end

  defp infer_confidence_from_score(score) when is_number(score) do
    max(0.0, min(1.0, score * 2.0 + 0.5))
  end

  defp infer_confidence_from_score(_), do: 0.5

  @doc """
  Sort by confidence, recompute best/second/margin/entropy from current confidences.
  Does not re-softmax scores (use `from_classifier` / `from_top_k` for that).
  """
  def normalize(%__MODULE__{} = l), do: finalize(l)

  defp finalize(%__MODULE__{candidates: []} = l), do: %{l | best: nil, second: nil, margin: 0.0, entropy: 0.0}

  defp finalize(%__MODULE__{candidates: [only]} = l) do
    c = %{only | confidence: clamp01(only.confidence)}
    %{l | candidates: [c], best: c, second: nil, margin: 0.0, entropy: 0.0}
  end

  defp finalize(%__MODULE__{candidates: cands} = l) do
    sorted = Enum.sort_by(cands, fn c -> {c.confidence, c.score} end, :desc)
    [best | rest] = sorted
    second = List.first(rest)

    margin =
      if second do
        best.confidence - second.confidence
      else
        0.0
      end

    sum = sorted |> Enum.map(& &1.confidence) |> Enum.sum()

    probs =
      if sum > 0.0 do
        Enum.map(sorted, &(&1.confidence / sum))
      else
        n = length(sorted)
        List.duplicate(1.0 / n, n)
      end

    ent = normalized_entropy(probs)

    %{l | candidates: sorted, best: best, second: second, margin: margin, entropy: ent}
  end

  defp clamp01(x) when is_number(x), do: max(0.0, min(1.0, x))
  defp clamp01(_), do: 0.5

  defp softmax(scores) do
    case scores do
      [] ->
        []

      [_single] ->
        [1.0]

      _ ->
        max_s = Enum.max(scores)
        exps = Enum.map(scores, fn s -> :math.exp(s - max_s) end)
        sum = Enum.sum(exps)

        if sum == 0.0 do
          n = length(scores)
          List.duplicate(1.0 / n, n)
        else
          Enum.map(exps, &(&1 / sum))
        end
    end
  end

  defp normalized_entropy(probs) when is_list(probs) do
    probs = Enum.filter(probs, &(&1 > 0.0))
    n = length(probs)

    cond do
      n <= 1 ->
        0.0

      true ->
        h = -Enum.reduce(probs, 0.0, fn p, acc -> acc + p * :math.log(p) end)
        h_max = :math.log(n * 1.0)
        if h_max > 0.0, do: min(1.0, max(0.0, -h / h_max)), else: 0.0
    end
  end

  def rerank(%__MODULE__{} = l, delta_fn) when is_function(delta_fn, 1) do
    updated =
      Enum.map(l.candidates, fn c ->
        d = delta_fn.(c) * 1.0
        %{c | confidence: clamp01(c.confidence + d)}
      end)

    finalize(%{l | candidates: updated})
  end

  def filter(%__MODULE__{} = l, pred) when is_function(pred, 1) do
    finalize(%{l | candidates: Enum.filter(l.candidates, pred)})
  end

  def map_labels(%__MODULE__{} = l, fun) when is_function(fun, 1) do
    finalize(%{
      l
      | candidates:
          Enum.map(l.candidates, fn c ->
            %{c | label: fun.(c.label)}
          end)
    })
  end

  def take_top_k(%__MODULE__{} = l, k) when is_integer(k) and k >= 0 do
    finalize(%{l | candidates: Enum.take(l.candidates, k)})
  end

  def merge(%__MODULE__{} = a, %__MODULE__{} = b, mode \\ :max) when mode in @merge_modes do
    merged =
      case mode do
        :max ->
          by_label =
            Enum.reduce(a.candidates ++ b.candidates, %{}, fn c, acc ->
              key = label_key(c.label)
              prev = Map.get(acc, key)

              cond do
                prev == nil ->
                  Map.put(acc, key, c)

                c.confidence > prev.confidence ->
                  Map.put(acc, key, c)

                true ->
                  acc
              end
            end)

          Map.values(by_label)

        :weighted_sum ->
          by_label =
            Enum.reduce(a.candidates ++ b.candidates, %{}, fn c, acc ->
              key = label_key(c.label)
              Map.update(acc, key, {c.label, c.confidence}, fn {lab, conf} -> {lab, conf + c.confidence} end)
            end)

          Enum.map(by_label, fn {_k, {lab, conf}} ->
            %Candidate{
              label: lab,
              score: conf,
              confidence: clamp01(conf / 2.0),
              source: :composite,
              metadata: %{}
            }
          end)

        :log_odds ->
          by_label =
            Enum.reduce(a.candidates ++ b.candidates, %{}, fn c, acc ->
              key = label_key(c.label)
              p = clamp01(c.confidence)
              lo = :math.log(p / (1.0 - p + 1.0e-9))
              Map.update(acc, key, {c.label, lo}, fn {lab, acc_lo} -> {lab, acc_lo + lo} end)
            end)

          Enum.map(by_label, fn {_k, {lab, lo}} ->
            conf = 1.0 / (1.0 + :math.exp(-lo))

            %Candidate{
              label: lab,
              score: lo,
              confidence: clamp01(conf),
              source: :composite,
              metadata: %{}
            }
          end)
      end

    finalize(%__MODULE__{
      candidates: merged,
      source: :composite,
      stage: a.stage || b.stage,
      error: nil
    })
  end

  defp label_key(l) when is_binary(l), do: l
  defp label_key(l) when is_atom(l), do: to_string(l)
  defp label_key(l), do: :erlang.phash2(l)
end
