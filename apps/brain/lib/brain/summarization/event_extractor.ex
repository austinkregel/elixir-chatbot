defmodule Brain.Summarization.EventExtractor do
  @moduledoc """
  Extracts semantic events from dialogue turns.

  Uses speech act classification to identify:
  - Assertives: Statements of fact, announcements
  - Commissives: Plans, promises, commitments
  - Expressives: Emotions, greetings, attitudes
  - Directives: Questions, requests

  Each turn may produce multiple events.
  """

  alias Brain.Summarization.Types.{Turn, Event}
  alias Brain.ML.Tokenizer

  # Life event keywords (high importance)
  @life_events ~w(
    promoted graduated married engaged divorced
    hired fired quit retired born died
    moved pregnant expecting baby
    won lost bought sold
  )

  # Plan/commitment keywords
  @plan_keywords ~w(
    will going gonna shall would could
    plan planning meet meeting visit visiting
    come coming go going leave leaving
    try trying want need
  )

  # State keywords
  @state_keywords ~w(
    is am are was were been being
    feel feeling felt looks looking seems
    happy sad angry tired excited worried
  )

  # Phatic/greeting keywords (skip in summary)
  @phatic_keywords ~w(
    hey hi hello bye goodbye thanks thank
    please sorry okay ok yeah yes no
    haha lol omg wow hmm
  )

  @doc """
  Extract events from a single turn.
  """
  @spec extract(Turn.t()) :: [Event.t()]
  def extract(%Turn{} = turn) do
    text = turn.text
    sentences = split_into_sentences(text)

    sentences
    |> Enum.flat_map(fn sentence ->
      extract_from_sentence(sentence, turn)
    end)
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  Extract events from all turns.
  """
  @spec extract_all([Turn.t()]) :: [Event.t()]
  def extract_all(turns) when is_list(turns) do
    Enum.flat_map(turns, &extract/1)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp split_into_sentences(text) do
    # Split on sentence boundaries while preserving content
    text
    |> String.split(~r/(?<=[.!?])\s+/)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end

  defp extract_from_sentence(sentence, turn) do
    tokens = tokenize_to_strings(sentence)
    tokens_lower = Enum.map(tokens, &String.downcase/1)

    # Determine event type and extract predicate
    cond do
      is_phatic?(tokens_lower) ->
        # Skip phatic utterances or mark them
        [create_phatic_event(turn, sentence)]

      is_question?(sentence, tokens_lower) ->
        [create_question_event(turn, sentence, tokens)]

      has_life_event?(tokens_lower) ->
        [create_life_event(turn, sentence, tokens, tokens_lower)]

      has_plan?(tokens_lower) ->
        [create_plan_event(turn, sentence, tokens, tokens_lower)]

      has_state?(tokens_lower) ->
        [create_state_event(turn, sentence, tokens, tokens_lower)]

      true ->
        # Default to assertive statement
        [create_statement_event(turn, sentence, tokens)]
    end
  end

  defp tokenize_to_strings(text) do
    Tokenizer.tokenize(text)
    |> Enum.map(fn
      %{text: t} -> t
      t when is_binary(t) -> t
      _ -> ""
    end)
    |> Enum.reject(&(&1 == ""))
  end

  defp is_phatic?(tokens_lower) do
    # Check if most tokens are phatic
    phatic_count = Enum.count(tokens_lower, &(&1 in @phatic_keywords))
    total = length(tokens_lower)

    total > 0 and phatic_count / total > 0.5
  end

  defp is_question?(sentence, _tokens_lower) do
    String.ends_with?(sentence, "?")
  end

  defp has_life_event?(tokens_lower) do
    Enum.any?(tokens_lower, &(&1 in @life_events))
  end

  defp has_plan?(tokens_lower) do
    Enum.any?(tokens_lower, &(&1 in @plan_keywords))
  end

  defp has_state?(tokens_lower) do
    # Check for copula + state pattern
    has_copula = Enum.any?(tokens_lower, &(&1 in ~w(is am are was were feel feeling)))
    has_state_word = Enum.any?(tokens_lower, &(&1 in @state_keywords))
    has_copula and has_state_word
  end

  defp create_phatic_event(turn, sentence) do
    Event.new(
      turn.index,
      turn.speaker,
      :phatic,
      sentence,
      sentence,
      subtype: detect_phatic_subtype(sentence)
    )
  end

  defp detect_phatic_subtype(sentence) do
    lower = String.downcase(sentence)
    cond do
      String.contains?(lower, ["hi", "hey", "hello"]) -> :greeting
      String.contains?(lower, ["bye", "goodbye", "see you"]) -> :farewell
      String.contains?(lower, ["thank", "thanks"]) -> :thanks
      String.contains?(lower, ["sorry"]) -> :apology
      true -> :other
    end
  end

  defp create_question_event(turn, sentence, tokens) do
    # Extract the question content
    predicate = extract_question_content(tokens)

    Event.new(
      turn.index,
      turn.speaker,
      :directive,
      predicate,
      sentence,
      subtype: :question
    )
  end

  defp extract_question_content(tokens) do
    # Remove question words and return the rest
    question_words = ~w(what where when why how who which do does did can could will would)

    tokens
    |> Enum.reject(fn t -> String.downcase(t) in question_words end)
    |> Enum.join(" ")
  end

  defp create_life_event(turn, sentence, tokens, tokens_lower) do
    # Find the life event keyword
    event_word = Enum.find(@life_events, fn kw -> kw in tokens_lower end)

    # For life events, create a clean predicate focused on the event
    predicate = create_life_event_predicate(event_word, tokens, tokens_lower)

    Event.new(
      turn.index,
      turn.speaker,
      :assertive,
      predicate,
      sentence,
      subtype: :announcement,
      subject: turn.speaker,  # Life events are about the speaker
      tense: detect_tense(tokens_lower)
    )
  end

  defp create_life_event_predicate(event_word, tokens, tokens_lower) do
    # Build a clean predicate around the life event
    event_idx = Enum.find_index(tokens_lower, &(&1 == event_word))
    
    if event_idx do
      # Take 2 words before and after the event word for context
      start_idx = max(0, event_idx - 1)
      end_idx = min(length(tokens) - 1, event_idx + 2)
      
      tokens
      |> Enum.slice(start_idx..end_idx)
      |> Enum.reject(&is_pronoun_word?/1)
      |> Enum.join(" ")
    else
      event_word
    end
  end

  defp is_pronoun_word?(word) do
    String.downcase(word) in ~w(i me my we our you your he she it they them his her its their)
  end

  defp create_plan_event(turn, sentence, tokens, tokens_lower) do
    # Find the plan keyword
    plan_word = Enum.find(@plan_keywords, fn kw -> kw in tokens_lower end)

    # Extract the action being planned (cleaned)
    predicate = extract_plan_predicate(tokens, tokens_lower, plan_word)
    |> clean_predicate_text()

    Event.new(
      turn.index,
      turn.speaker,
      :commissive,
      predicate,
      sentence,
      subtype: :plan,
      subject: turn.speaker,
      tense: :future
    )
  end

  defp clean_predicate_text(text) do
    # Remove pronouns and clean up
    text
    |> String.split()
    |> Enum.reject(&is_pronoun_word?/1)
    |> Enum.join(" ")
    |> String.trim()
  end

  defp create_state_event(turn, sentence, tokens, tokens_lower) do
    # Extract the state description
    state = extract_state(tokens, tokens_lower)

    Event.new(
      turn.index,
      turn.speaker,
      :assertive,
      state,
      sentence,
      subtype: :state,
      subject: turn.speaker,
      tense: :present
    )
  end

  defp create_statement_event(turn, sentence, tokens) do
    # General statement - extract main content
    predicate = Enum.join(tokens, " ")

    Event.new(
      turn.index,
      turn.speaker,
      :assertive,
      predicate,
      sentence,
      subtype: :statement
    )
  end

  defp extract_plan_predicate(tokens, tokens_lower, plan_word) do
    # Find position of plan keyword
    plan_idx = Enum.find_index(tokens_lower, &(&1 == plan_word))

    if plan_idx do
      # Take everything after the plan word
      tokens
      |> Enum.drop(plan_idx)
      |> Enum.join(" ")
    else
      Enum.join(tokens, " ")
    end
  end

  defp extract_state(tokens, tokens_lower) do
    # Find copula and take what follows
    copula_words = ~w(is am are was were feel feeling)
    copula_idx = Enum.find_index(tokens_lower, &(&1 in copula_words))

    if copula_idx do
      tokens
      |> Enum.drop(copula_idx)
      |> Enum.join(" ")
    else
      Enum.join(tokens, " ")
    end
  end

  defp detect_tense(tokens_lower) do
    cond do
      Enum.any?(tokens_lower, &(&1 in ~w(will going gonna shall))) -> :future
      Enum.any?(tokens_lower, &(&1 in ~w(was were did had been))) -> :past
      true -> :present
    end
  end
end
