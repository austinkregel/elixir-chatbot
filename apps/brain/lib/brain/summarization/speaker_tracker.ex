defmodule Brain.Summarization.SpeakerTracker do
  @moduledoc """
  Tracks speakers and builds pronoun mappings across dialogue turns.

  For each turn, determines:
  - Who is the current speaker
  - Who is being addressed (if any)
  - How to map pronouns to concrete names

  Uses patterns from DiscourseAnalyzer for pronoun detection.
  """

  alias Brain.Summarization.Types.{Turn, SpeakerContext}
  alias Brain.ML.Tokenizer

  # Pronoun categories
  @first_person_singular ~w(i me my mine myself)
  @first_person_plural ~w(we us our ours ourselves)
  @second_person ~w(you your yours yourself yourselves)
  @third_person_singular ~w(he she it him her his hers its)
  @third_person_plural ~w(they them their theirs themselves)

  @doc """
  Build speaker context across all turns.

  Returns a map of turn_index -> SpeakerContext with pronoun mappings
  for that turn.
  """
  @spec track([Turn.t()]) :: %{non_neg_integer() => SpeakerContext.t()}
  def track(turns) when is_list(turns) do
    participants = extract_all_participants(turns)
    initial_context = SpeakerContext.new(participants)

    {contexts, _final_ctx} = 
      Enum.reduce(turns, {%{}, initial_context}, fn turn, {contexts, ctx} ->
        # Determine who is being addressed
        addressee = infer_addressee(turn, participants)

        # Update context for this turn
        updated_ctx = SpeakerContext.update_for_turn(ctx, turn.speaker, addressee)

        # Extract any mentioned entities for later resolution
        updated_ctx = extract_mentioned_entities(updated_ctx, turn)

        {Map.put(contexts, turn.index, updated_ctx), updated_ctx}
      end)

    contexts
  end

  @doc """
  Resolve a pronoun to a concrete name given the turn context.
  """
  @spec resolve_pronoun(String.t(), SpeakerContext.t()) :: String.t() | nil
  def resolve_pronoun(pronoun, %SpeakerContext{} = ctx) do
    pronoun_lower = String.downcase(pronoun)

    cond do
      # First person -> current speaker
      pronoun_lower in @first_person_singular ->
        ctx.current_speaker

      # Second person -> addressee or other participant
      pronoun_lower in @second_person ->
        ctx.addressee || other_participant(ctx)

      # First person plural -> speaker + others
      pronoun_lower in @first_person_plural ->
        ctx.current_speaker

      # Third person -> most recently mentioned entity of matching gender/type
      pronoun_lower in @third_person_singular ->
        resolve_third_person(pronoun_lower, ctx)

      pronoun_lower in @third_person_plural ->
        resolve_third_person_plural(ctx)

      # Direct lookup
      true ->
        Map.get(ctx.pronoun_map, pronoun_lower)
    end
  end

  @doc """
  Check if a word is a pronoun.
  """
  @spec is_pronoun?(String.t()) :: boolean()
  def is_pronoun?(word) do
    word_lower = String.downcase(word)
    word_lower in (@first_person_singular ++ @first_person_plural ++
                   @second_person ++ @third_person_singular ++ @third_person_plural)
  end

  @doc """
  Get the pronoun type for categorization.
  """
  @spec pronoun_type(String.t()) :: atom() | nil
  def pronoun_type(word) do
    word_lower = String.downcase(word)

    cond do
      word_lower in @first_person_singular -> :first_singular
      word_lower in @first_person_plural -> :first_plural
      word_lower in @second_person -> :second
      word_lower in @third_person_singular -> :third_singular
      word_lower in @third_person_plural -> :third_plural
      true -> nil
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp extract_all_participants(turns) do
    turns
    |> Enum.map(& &1.speaker)
    |> Enum.uniq()
  end

  defp infer_addressee(turn, participants) do
    text_lower = String.downcase(turn.text)

    # Check if turn directly addresses someone by name
    other_participants = Enum.reject(participants, &(&1 == turn.speaker))

    addressed = Enum.find(other_participants, fn participant ->
      name_lower = String.downcase(participant)
      # Check for direct address patterns
      String.contains?(text_lower, name_lower <> ",") or
      String.starts_with?(text_lower, name_lower) or
      String.contains?(text_lower, "@" <> name_lower)
    end)

    # If not explicitly addressed, infer from context
    addressed || infer_default_addressee(turn, other_participants)
  end

  defp infer_default_addressee(_turn, []), do: nil
  defp infer_default_addressee(_turn, [single]), do: single
  defp infer_default_addressee(turn, participants) do
    # In multi-party dialogue, use "you" patterns to infer
    text_lower = String.downcase(turn.text)

    if String.contains?(text_lower, "you") do
      # Default to first other participant
      List.first(participants)
    else
      nil
    end
  end

  defp other_participant(%SpeakerContext{participants: participants, current_speaker: speaker}) do
    participants
    |> Enum.reject(&(&1 == speaker))
    |> List.first()
  end

  defp extract_mentioned_entities(ctx, turn) do
    tokens = Tokenizer.tokenize(turn.text)

    # Look for capitalized words that might be entity mentions
    entities = 
      tokens
      |> Enum.map(&get_token_text/1)
      |> Enum.filter(&is_potential_entity?/1)
      |> Enum.reject(fn name -> name == turn.speaker end)
      |> Enum.map(fn text ->
        %{text: text, type: :unknown, turn: turn.index}
      end)

    %{ctx | mentioned_entities: entities ++ ctx.mentioned_entities}
  end

  defp get_token_text(%{text: text}), do: text
  defp get_token_text(text) when is_binary(text), do: text
  defp get_token_text(_), do: ""

  defp is_potential_entity?(text) when is_binary(text) do
    String.length(text) >= 2 and
    String.at(text, 0) == String.upcase(String.at(text, 0)) and
    String.at(text, 0) =~ ~r/[A-Z]/
  end
  defp is_potential_entity?(_), do: false

  defp resolve_third_person(_pronoun, ctx) do
    # For "he"/"him"/"his" -> look for masculine names
    # For "she"/"her"/"hers" -> look for feminine names
    # For "it"/"its" -> look for objects

    case ctx.mentioned_entities do
      [] -> nil
      [recent | _] -> recent.text
    end
  end

  defp resolve_third_person_plural(ctx) do
    case ctx.participants do
      [] -> nil
      participants when length(participants) > 1 ->
        Enum.join(participants, " and ")
      [single] -> single
    end
  end
end
