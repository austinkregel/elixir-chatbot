defmodule Brain.Summarization.ReferenceResolver do
  @moduledoc """
  Resolves pronoun and anaphoric references in events.

  Takes events with unresolved pronouns (e.g., "I got promoted") and
  resolves them to concrete names (e.g., "Demi got promoted").

  Uses speaker context from SpeakerTracker for resolution.
  """

  alias Brain.Summarization.Types.{Event, SpeakerContext}
  alias Brain.Summarization.SpeakerTracker
  alias Brain.ML.Tokenizer

  @doc """
  Resolve all pronoun references in a list of events.

  Uses the speaker contexts (keyed by turn index) to resolve pronouns.
  """
  @spec resolve([Event.t()], %{non_neg_integer() => SpeakerContext.t()}) :: [Event.t()]
  def resolve(events, speaker_contexts) when is_list(events) and is_map(speaker_contexts) do
    Enum.map(events, fn event ->
      ctx = Map.get(speaker_contexts, event.turn_index, %SpeakerContext{})
      resolve_event(event, ctx)
    end)
  end

  @doc """
  Resolve pronouns in a single event.
  """
  @spec resolve_event(Event.t(), SpeakerContext.t()) :: Event.t()
  def resolve_event(%Event{} = event, %SpeakerContext{} = ctx) do
    # Resolve subject if it's a pronoun
    resolved_subject = resolve_if_pronoun(event.subject, ctx, event.speaker)

    # Resolve object if it's a pronoun
    resolved_object = resolve_if_pronoun(event.object, ctx, event.speaker)

    # Resolve pronouns in the predicate text
    resolved_predicate = resolve_predicate_pronouns(event.predicate, ctx, event.speaker)

    %{event |
      subject: resolved_subject,
      object: resolved_object,
      predicate: resolved_predicate
    }
  end

  @doc """
  Resolve a single text string, replacing all pronouns with names.
  """
  @spec resolve_text(String.t(), SpeakerContext.t(), String.t()) :: String.t()
  def resolve_text(text, ctx, speaker) when is_binary(text) do
    resolve_predicate_pronouns(text, ctx, speaker)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp resolve_if_pronoun(nil, _ctx, speaker), do: speaker
  defp resolve_if_pronoun(text, ctx, speaker) when is_binary(text) do
    if SpeakerTracker.is_pronoun?(text) do
      SpeakerTracker.resolve_pronoun(text, ctx) || speaker
    else
      text
    end
  end

  defp resolve_predicate_pronouns(nil, _ctx, _speaker), do: nil
  defp resolve_predicate_pronouns(predicate, ctx, speaker) when is_binary(predicate) do
    tokens = tokenize_to_strings(predicate)

    # Only resolve subject pronouns (first occurrence of I/we/you at sentence start)
    # Don't replace every pronoun - that makes text unreadable
    {resolved_tokens, _replaced} = 
      Enum.reduce(tokens, {[], false}, fn token, {acc, already_replaced} ->
        token_lower = String.downcase(token)
        
        # Only replace subject pronouns, and only once per predicate
        is_subject_pronoun = token_lower in ~w(i we)
        should_replace = is_subject_pronoun and not already_replaced
        
        if should_replace and SpeakerTracker.is_pronoun?(token) do
          resolved = SpeakerTracker.resolve_pronoun(token, ctx)
          adapted = adapt_case(resolved || speaker, token)
          {acc ++ [adapted], true}
        else
          {acc ++ [token], already_replaced}
        end
      end)

    join_tokens(resolved_tokens)
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

  defp adapt_case(resolved, original) do
    # Always use proper name capitalization, not the pronoun's case
    # "I" should become "Demi", not "DEMI"
    cond do
      original == "I" ->
        # Special case: "I" is always capitalized but name should be normal
        String.capitalize(resolved)

      String.at(original, 0) == String.upcase(String.at(original, 0)) ->
        # Title Case
        String.capitalize(resolved)

      true ->
        # lowercase context - still capitalize the name
        String.capitalize(resolved)
    end
  end

  defp join_tokens(tokens) do
    # Rejoin tokens preserving spacing around punctuation
    tokens
    |> Enum.reduce({"", false}, fn token, {acc, needs_space} ->
      is_punct = token in ~w(. , ! ? ; : ' ")

      cond do
        acc == "" ->
          {token, true}
        is_punct ->
          {acc <> token, true}
        needs_space ->
          {acc <> " " <> token, true}
        true ->
          {acc <> token, true}
      end
    end)
    |> elem(0)
  end
end
