defmodule Brain.Summarization.Pipeline do
  @moduledoc """
  Orchestrates the dialogue summarization pipeline.

  Pipeline steps:
  1. Turn Segmenter - Parse dialogue into turns
  2. Speaker Tracker - Build pronoun mappings
  3. Event Extractor - Extract semantic events
  4. Reference Resolver - Resolve pronouns to names
  5. Fact Ranker - Rank by importance
  6. Summary Builder - Generate summary text

  ## Example

      iex> dialogue = "Lucas: Hey!, Demi: I got promoted! :D, Lucas: Congratulations!"
      iex> {:ok, summary} = Pipeline.summarize(dialogue)
      iex> summary.text
      "Demi got promoted."
  """

  alias Brain.Summarization.{
    TurnSegmenter,
    SpeakerTracker,
    EventExtractor,
    ReferenceResolver,
    FactRanker,
    SummaryBuilder
  }
  alias Brain.Summarization.Types.Summary

  require Logger

  @doc """
  Run the full summarization pipeline on dialogue text.

  ## Options
    - :max_facts - Maximum number of facts to include (default: 3)
    - :verbose - Log intermediate steps (default: false)
  """
  @spec summarize(String.t(), keyword()) :: {:ok, Summary.t()} | {:error, term()}
  def summarize(dialogue, opts \\ []) when is_binary(dialogue) do
    verbose = Keyword.get(opts, :verbose, false)
    max_facts = Keyword.get(opts, :max_facts, 3)

    try do
      # Step 1: Segment into turns
      {:ok, turns} = TurnSegmenter.segment(dialogue)
      if verbose, do: log_step("Segmented", "#{length(turns)} turns")

      if Enum.empty?(turns) do
        {:ok, Summary.new("No dialogue content found.")}
      else
        # Step 2: Track speakers and build pronoun maps
        speaker_contexts = SpeakerTracker.track(turns)
        participants = TurnSegmenter.extract_participants(dialogue)
        if verbose, do: log_step("Tracked", "#{length(participants)} participants: #{inspect(participants)}")

        # Step 3: Extract events from turns
        events = EventExtractor.extract_all(turns)
        if verbose, do: log_step("Extracted", "#{length(events)} events")

        # Step 4: Resolve pronoun references
        resolved_events = ReferenceResolver.resolve(events, speaker_contexts)
        if verbose, do: log_step("Resolved", "pronouns in #{length(resolved_events)} events")

        # Step 5: Rank facts by importance
        ranked_facts = FactRanker.rank(resolved_events)
        if verbose, do: log_step("Ranked", "#{length(ranked_facts)} facts, top score: #{top_score(ranked_facts)}")

        # Step 6: Build summary
        summary = SummaryBuilder.build(ranked_facts,
          max_facts: max_facts,
          participants: participants
        )
        if verbose, do: log_step("Built", "summary: #{summary.text}")

        {:ok, summary}
      end
    rescue
      e ->
        Logger.error("Summarization failed: #{inspect(e)}")
        {:error, {:summarization_failed, e}}
    end
  end

  @doc """
  Get just the summary text (convenience function).
  """
  @spec summarize_text(String.t(), keyword()) :: String.t()
  def summarize_text(dialogue, opts \\ []) do
    case summarize(dialogue, opts) do
      {:ok, summary} -> summary.text
      {:error, _} -> "Unable to summarize dialogue."
    end
  end

  @doc """
  Run pipeline with detailed trace for debugging.
  """
  @spec summarize_with_trace(String.t()) :: map()
  def summarize_with_trace(dialogue) do
    {:ok, turns} = TurnSegmenter.segment(dialogue)
    speaker_contexts = SpeakerTracker.track(turns)
    events = EventExtractor.extract_all(turns)
    resolved_events = ReferenceResolver.resolve(events, speaker_contexts)
    ranked_facts = FactRanker.rank(resolved_events)

    participants = TurnSegmenter.extract_participants(dialogue)
    summary = SummaryBuilder.build(ranked_facts, participants: participants)

    %{
      input: dialogue,
      turns: Enum.map(turns, fn t -> %{speaker: t.speaker, text: t.text} end),
      participants: participants,
      events: Enum.map(events, &event_to_map/1),
      resolved_events: Enum.map(resolved_events, &event_to_map/1),
      ranked_facts: Enum.map(ranked_facts, &fact_to_map/1),
      summary: summary.text,
      sentences: summary.sentences
    }
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp log_step(step, message) do
    Logger.debug("[Summarization] #{step}: #{message}")
  end

  defp top_score([]), do: 0.0
  defp top_score([first | _]), do: Float.round(first.score, 2)

  defp event_to_map(event) do
    %{
      speaker: event.speaker,
      type: event.type,
      subtype: event.subtype,
      subject: event.subject,
      predicate: event.predicate,
      raw_text: String.slice(event.raw_text, 0, 50)
    }
  end

  defp fact_to_map(fact) do
    %{
      subject: fact.subject,
      predicate: String.slice(fact.predicate, 0, 50),
      importance: fact.importance,
      score: Float.round(fact.score, 2)
    }
  end
end
