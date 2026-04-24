defmodule Brain.Analysis.ChunkPriority do
  @moduledoc """
  Selects the "primary" `ChunkAnalysis` from a list of per-chunk analyses for
  downstream response grounding (content specification, surface realization,
  knowledge lookup).

  Selection is driven entirely by signals already produced by upstream
  analyzers (`SpeechActClassifier`, `EntityExtractor`) - no hardcoded intent
  strings, no regex on text. The intent is to honor what analysis already
  knows: a clearly-marked question/directive should outrank a higher-confidence
  but non-actionable assertion (e.g. a meta classification on an introductory
  sentence).

  ## Priority order

    1. Chunk whose `speech_act.is_question == true` AND
       `speech_act.category == :directive` (an explicit ask)
    2. Chunk whose `speech_act.category == :directive` (any sub-type)
    3. Chunk whose `speech_act.category == :assertive` AND has at least one
       extracted entity
    4. Chunk with the highest `confidence` value (legacy fallback)
    5. First chunk in the list

  When several chunks are tied at the same priority tier, the higher-`confidence`
  chunk wins; on a further tie, the earlier `chunk_index` wins.
  """

  alias Brain.Analysis.ChunkAnalysis

  @type analysis :: ChunkAnalysis.t() | map()

  @doc """
  Returns the primary analysis from `analyses`.

  Returns an empty `%ChunkAnalysis{}` for an empty list, mirroring the
  behavior previously inlined in `RefinementLoop.select_primary_analysis/1`.
  """
  @spec select_primary([analysis()]) :: analysis()
  def select_primary([]), do: %ChunkAnalysis{}

  def select_primary(analyses) when is_list(analyses) do
    analyses
    |> Enum.with_index()
    |> Enum.max_by(fn {analysis, idx} -> sort_key(analysis, idx) end)
    |> elem(0)
  end

  @doc """
  Returns the analysis the system should treat as the actionable question /
  request, if there is one.

  Unlike `select_primary/1`, this returns `nil` when no chunk looks like a
  user-facing question or directive. Callers (e.g. knowledge lookup routing)
  use this to decide whether to fan out fact retrieval to a chunk other than
  the primary.
  """
  @spec question_chunk([analysis()]) :: analysis() | nil
  def question_chunk(analyses) when is_list(analyses) do
    Enum.find(analyses, &question?/1) ||
      Enum.find(analyses, &directive?/1)
  end

  def question_chunk(_), do: nil

  defp sort_key(analysis, idx) do
    tier =
      cond do
        question?(analysis) -> 4
        directive?(analysis) -> 3
        assertive_with_entities?(analysis) -> 2
        true -> 1
      end

    confidence = confidence_of(analysis)

    {tier, confidence, -idx}
  end

  defp question?(analysis) do
    speech_act = speech_act_of(analysis)

    Map.get(speech_act, :is_question, false) == true and
      Map.get(speech_act, :category) == :directive
  end

  defp directive?(analysis) do
    speech_act_of(analysis) |> Map.get(:category) == :directive
  end

  defp assertive_with_entities?(analysis) do
    speech_act_of(analysis) |> Map.get(:category) == :assertive and
      entity_count(analysis) > 0
  end

  defp speech_act_of(%{speech_act: nil}), do: %{}
  defp speech_act_of(%{speech_act: speech_act}) when is_map(speech_act), do: speech_act
  defp speech_act_of(_), do: %{}

  defp entity_count(analysis) do
    case Map.get(analysis, :entities) do
      list when is_list(list) -> length(list)
      _ -> 0
    end
  end

  defp confidence_of(analysis) do
    case Map.get(analysis, :confidence) do
      conf when is_number(conf) -> conf
      _ -> 0.0
    end
  end
end
