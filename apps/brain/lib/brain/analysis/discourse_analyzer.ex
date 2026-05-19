defmodule Brain.Analysis.DiscourseAnalyzer do
  @moduledoc """
  Analyzes discourse structure to determine who is being addressed.

  This module detects:
  - Direct address patterns ("Hey bot", "Companion,")
  - Second person pronouns ("you", "your")
  - Imperative mood (commands directed at someone)
  - Question patterns (typically expect a response)
  - Implicit addressee (in 1-on-1, defaults to bot)
  """

  alias Brain.Analysis.DiscourseResult
  alias Brain.ML.Tokenizer

  require Logger

  # Load discourse configuration from JSON at compile time
  @discourse_config_path "priv/analysis/discourse_config.json"
  @external_resource @discourse_config_path

  @discourse_config (case File.read(@discourse_config_path) do
                       {:ok, content} ->
                         case Jason.decode(content) do
                           {:ok, data} -> data
                           {:error, _} -> %{}
                         end

                       {:error, _} ->
                         %{}
                     end)

  # Bot name patterns - loaded from discourse_config.json
  @default_bot_names Map.get(@discourse_config, "bot_names", ["companion", "bot", "assistant", "ai", "echo"])

  # Second person pronouns indicating addressing someone
  @second_person_pronouns ~w(you your yours yourself)

  # First person pronouns (user speaking about themselves)
  @first_person_pronouns ~w(i me my mine myself we us our ours ourselves)

  # Third person pronouns (talking about someone else)
  @third_person_pronouns ~w(he she it they him her them his hers its their theirs)

  # Direct address indicators - loaded from discourse_config.json
  @address_prefixes Map.get(@discourse_config, "address_prefixes", ["hey", "hi", "hello", "ok", "okay", "yo"])

  @doc """
  Analyzes discourse structure to determine the addressee.

  Options:
  - :participants - list of conversation participants (default: [:user, :bot])
  - :bot_names - additional names the bot responds to

  Returns a DiscourseResult struct.
  """
  def analyze(text, opts \\ []) do
    participants = Keyword.get(opts, :participants, [:user, :bot])
    extra_bot_names = Keyword.get(opts, :bot_names, [])
    bot_names = @default_bot_names ++ Enum.map(extra_bot_names, &String.downcase/1)

    normalized = normalize_text(text)

    # Collect all indicators and scores
    {indicators, scores} = analyze_patterns(normalized, text, bot_names, participants)

    # Determine addressee based on scores
    {addressee, confidence} = determine_addressee(scores, participants)

    DiscourseResult.new(addressee, confidence, indicators)
    |> Map.put(:participants, participants)
  end

  @doc """
  Returns detailed analysis for debugging/learning.
  """
  def debug_analyze(text, opts \\ []) do
    result = analyze(text, opts)
    normalized = normalize_text(text)

    %{
      result: result,
      normalized_text: normalized,
      detected_pronouns: detect_pronouns(normalized),
      has_direct_address: has_direct_address?(normalized, @default_bot_names),
      has_imperative: has_imperative_start?(normalized),
      is_question: is_question?(text)
    }
  end

  # Private functions

  defp normalize_text(text) do
    # Use Tokenizer for regex-free normalization
    Tokenizer.normalize(text)
  end

  defp analyze_patterns(normalized, original, bot_names, participants) do
    indicators = []
    scores = %{bot: 0.0, user: 0.0, third_party: 0.0, ambiguous: 0.0}

    # Check for direct address to bot
    {indicators, scores} =
      if has_direct_address?(normalized, bot_names) do
        {["direct_address" | indicators], Map.update!(scores, :bot, &(&1 + 0.5))}
      else
        {indicators, scores}
      end

    # Check for second person pronouns
    {indicators, scores} =
      if has_second_person?(normalized) do
        new_indicators = ["second_person_pronoun" | indicators]

        # In 1-on-1 with bot, "you" refers to bot
        new_scores =
          if :bot in participants and length(participants) == 2 do
            Map.update!(scores, :bot, &(&1 + 0.3))
          else
            Map.update!(scores, :ambiguous, &(&1 + 0.2))
          end

        {new_indicators, new_scores}
      else
        {indicators, scores}
      end

    # Check for imperative mood
    {indicators, scores} =
      if has_imperative_start?(normalized) do
        new_indicators = ["imperative_mood" | indicators]

        # Imperatives in 1-on-1 are directed at bot
        new_scores =
          if :bot in participants and length(participants) == 2 do
            Map.update!(scores, :bot, &(&1 + 0.25))
          else
            Map.update!(scores, :ambiguous, &(&1 + 0.15))
          end

        {new_indicators, new_scores}
      else
        {indicators, scores}
      end

    # Check for question (expects response from addressee)
    {indicators, scores} =
      if is_question?(original) do
        new_indicators = ["question" | indicators]

        # Questions in 1-on-1 expect bot to answer
        new_scores =
          if :bot in participants and length(participants) == 2 do
            Map.update!(scores, :bot, &(&1 + 0.2))
          else
            Map.update!(scores, :ambiguous, &(&1 + 0.1))
          end

        {new_indicators, new_scores}
      else
        {indicators, scores}
      end

    # Check for "can you", "could you", "would you" patterns
    {indicators, scores} =
      if has_modal_you_pattern?(normalized) do
        new_indicators = ["modal_you_request" | indicators]
        new_scores = Map.update!(scores, :bot, &(&1 + 0.35))
        {new_indicators, new_scores}
      else
        {indicators, scores}
      end

    # Check for self-referential statements (user talking about themselves)
    {indicators, scores} =
      if primarily_first_person?(normalized) and not has_second_person?(normalized) do
        new_indicators = ["self_referential" | indicators]
        # User making a statement, not necessarily addressing bot
        new_scores = Map.update!(scores, :ambiguous, &(&1 + 0.15))
        {new_indicators, new_scores}
      else
        {indicators, scores}
      end

    # Check for third person references
    {indicators, scores} =
      if primarily_third_person?(normalized) do
        new_indicators = ["third_person_reference" | indicators]
        new_scores = Map.update!(scores, :third_party, &(&1 + 0.2))
        {new_indicators, new_scores}
      else
        {indicators, scores}
      end

    # Baseline: in 1-on-1 conversation, default to bot addressee
    scores =
      if :bot in participants and length(participants) == 2 do
        Map.update!(scores, :bot, &(&1 + 0.1))
      else
        scores
      end

    {indicators, scores}
  end

  defp determine_addressee(scores, participants) do
    # Find highest scoring addressee
    {best_addressee, best_score} =
      scores
      |> Enum.filter(fn {addressee, _} ->
        case addressee do
          :bot -> :bot in participants
          :user -> :user in participants
          :third_party -> true
          :ambiguous -> true
        end
      end)
      |> Enum.max_by(fn {_, score} -> score end, fn -> {:unknown, 0.0} end)

    # Normalize confidence to 0-1 range
    total_score = Enum.reduce(scores, 0, fn {_, s}, acc -> acc + s end)

    confidence =
      if total_score > 0 do
        min(best_score / total_score + 0.3, 1.0)
      else
        0.5
      end

    # If ambiguous has the highest score but bot is close, prefer bot in 1-on-1
    final_addressee =
      cond do
        best_addressee == :ambiguous and :bot in participants and length(participants) == 2 ->
          bot_score = Map.get(scores, :bot, 0)
          if bot_score > 0.1, do: :bot, else: :ambiguous

        best_score < 0.1 ->
          :unknown

        true ->
          best_addressee
      end

    {final_addressee, confidence}
  end

  defp has_direct_address?(normalized, bot_names) do
    words = Tokenizer.tokenize_words(normalized)
    first_word = List.first(words) || ""

    structural_match =
      Enum.any?(bot_names, fn name ->
        name_lower = String.downcase(name)

        Enum.any?(@address_prefixes, fn prefix ->
          prefix_lower = String.downcase(prefix)
          idx = Enum.find_index(words, &(&1 == prefix_lower))
          idx != nil and Enum.at(words, idx + 1) == name_lower
        end) or
          first_word == name_lower or
          Enum.member?(words, "@" <> name_lower)
      end)

    classifier_match =
      case Brain.ML.MicroClassifiers.classify(:directed_at_bot, normalized) do
        {:ok, "directed", score} when score > 0.4 -> true
        _ -> false
      end

    structural_match or classifier_match
  end

  defp has_second_person?(normalized) do
    words = Tokenizer.split_words(normalized)
    Enum.any?(@second_person_pronouns, &(&1 in words))
  end

  defp primarily_first_person?(normalized) do
    words = Tokenizer.split_words(normalized)
    first_person_count = Enum.count(words, &(&1 in @first_person_pronouns))
    second_person_count = Enum.count(words, &(&1 in @second_person_pronouns))
    first_person_count > second_person_count and first_person_count > 0
  end

  defp primarily_third_person?(normalized) do
    words = Tokenizer.split_words(normalized)
    third_person_count = Enum.count(words, &(&1 in @third_person_pronouns))
    first_person_count = Enum.count(words, &(&1 in @first_person_pronouns))
    second_person_count = Enum.count(words, &(&1 in @second_person_pronouns))
    third_person_count > first_person_count and third_person_count > second_person_count
  end

  defp has_imperative_start?(normalized) do
    imperative_verbs = ~w(
      tell show give get find search look check
      turn set make create open close start stop
      play pause skip next list read send call
      help explain describe calculate remember
    )

    first_word = normalized |> Tokenizer.split_words() |> List.first() || ""
    first_word in imperative_verbs
  end

  defp has_modal_you_pattern?(normalized) do
    case Brain.ML.MicroClassifiers.classify(:modal_directive, normalized) do
      {:ok, "directive", score} when score > 0.3 -> true
      _ -> false
    end
  end

  defp is_question?(text) do
    Tokenizer.ends_with_question?(text)
  end

  defp detect_pronouns(normalized) do
    words = Tokenizer.split_words(normalized)

    %{
      first_person: Enum.filter(words, &(&1 in @first_person_pronouns)),
      second_person: Enum.filter(words, &(&1 in @second_person_pronouns)),
      third_person: Enum.filter(words, &(&1 in @third_person_pronouns))
    }
  end
end
