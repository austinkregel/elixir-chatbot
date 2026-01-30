defmodule ChatBot.Summarization.SummaryBuilder do
  @moduledoc """
  Builds coherent summary text from ranked facts.

  Uses template-based generation to produce grammatical sentences.
  Combines multiple facts into a flowing summary.
  """

  alias ChatBot.Summarization.Types.{Fact, Summary}

  # Templates for different fact types
  @templates %{
    life_event: [
      "{subject} {predicate}.",
      "{subject} has {predicate}."
    ],
    plan: [
      "{subject} will {predicate}.",
      "{subject} is going to {predicate}.",
      "{participants} will {predicate}."
    ],
    statement: [
      "{subject} {predicate}.",
      "{predicate}."
    ],
    state: [
      "{subject} is {predicate}.",
      "{subject} {predicate}."
    ],
    question: [
      "{subject} asked about {predicate}."
    ],
    generic: [
      "{subject} {predicate}."
    ]
  }

  @doc """
  Build a summary from ranked facts.

  Takes top N facts and generates coherent text.
  """
  @spec build([Fact.t()], keyword()) :: Summary.t()
  def build(facts, opts \\ []) do
    max_facts = Keyword.get(opts, :max_facts, 3)
    participants = Keyword.get(opts, :participants, [])

    # Take top facts
    top_facts = Enum.take(facts, max_facts)

    # Generate sentences for each fact
    sentences = 
      top_facts
      |> Enum.map(&fact_to_sentence/1)
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()

    # Combine into summary text
    text = Enum.join(sentences, " ")

    Summary.new(text,
      sentences: sentences,
      participants: participants,
      fact_count: length(top_facts),
      facts_used: top_facts
    )
  end

  @doc """
  Generate a single sentence from a fact.
  """
  @spec fact_to_sentence(Fact.t()) :: String.t() | nil
  def fact_to_sentence(%Fact{} = fact) do
    template_key = fact.template_key || :generic
    templates = Map.get(@templates, template_key, @templates.generic)

    # Choose best template based on available data
    template = choose_template(templates, fact)

    # Fill in the template
    sentence = fill_template(template, fact)

    # Clean up the result
    clean_sentence(sentence)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp choose_template(templates, fact) do
    # Prefer templates that use available data
    has_participants = length(fact.participants) > 1

    template = Enum.find(templates, fn t ->
      cond do
        has_participants and String.contains?(t, "{participants}") -> true
        String.contains?(t, "{subject}") and fact.subject != nil -> true
        true -> false
      end
    end)

    template || List.first(templates)
  end

  defp fill_template(template, fact) do
    subject = format_subject(fact.subject)
    predicate = format_predicate(fact.predicate, fact.tense, subject)
    
    template
    |> String.replace("{subject}", subject)
    |> String.replace("{predicate}", predicate)
    |> String.replace("{object}", fact.object || "")
    |> String.replace("{participants}", format_participants(fact.participants))
  end

  defp format_subject(nil), do: "Someone"
  defp format_subject(subject) do
    # Ensure proper capitalization
    String.capitalize(String.trim(subject))
  end

  defp format_predicate(nil, _tense, _subject), do: ""
  defp format_predicate(predicate, tense, subject) do
    cleaned = clean_predicate(predicate, subject)

    # Adjust verb tense if needed
    case tense do
      :past -> ensure_past_tense(cleaned)
      :future -> ensure_future_tense(cleaned)
      _ -> cleaned
    end
  end

  defp clean_predicate(predicate, subject) do
    predicate
    |> String.trim()
    |> remove_leading_pronouns()
    |> remove_subject_echo(subject)
    |> remove_trailing_punctuation_noise()
    |> String.trim()
  end

  defp remove_leading_pronouns(text) do
    pronouns = ~w(i me my we our you your he she it they them)

    words = String.split(text)

    case words do
      [first | rest] ->
        if String.downcase(first) in pronouns do
          Enum.join(rest, " ")
        else
          text
        end
      [] -> text
      _ -> text
    end
  end

  defp remove_subject_echo(text, subject) do
    # Remove the subject name if it appears at the start of the predicate
    # (since template already adds "{subject} {predicate}")
    subject_lower = String.downcase(subject)
    words = String.split(text)
    
    case words do
      [first | rest] ->
        if String.downcase(first) == subject_lower do
          Enum.join(rest, " ")
        else
          text
        end
      _ -> text
    end
  end

  defp remove_trailing_punctuation_noise(text) do
    # Remove trailing commas and extra punctuation
    text
    |> String.trim_trailing(",")
    |> String.trim_trailing("?")
    |> String.trim()
  end

  defp ensure_past_tense(text) do
    # Simple past tense handling - prepend "had" if not already past
    words = String.split(text)

    case words do
      [verb | _rest] ->
        verb_lower = String.downcase(verb)
        if past_tense_verb?(verb_lower) do
          text
        else
          # Check if we should add auxiliary
          if verb_lower in ~w(is are am was were has have had) do
            text
          else
            text
          end
        end
      _ -> text
    end
  end

  defp past_tense_verb?(verb) do
    String.ends_with?(verb, "ed") or
    verb in ~w(got went had was were did made said took came gave)
  end

  defp ensure_future_tense(text) do
    # Remove "will" or "going to" if already present
    cleaned = text
              |> String.replace(~r/^will\s+/i, "")
              |> String.replace(~r/^going to\s+/i, "")
              |> String.replace(~r/^gonna\s+/i, "")

    # Fix gerund after "will" - "visiting" -> "visit"
    fix_gerund_for_future(cleaned)
  end

  defp fix_gerund_for_future(text) do
    words = String.split(text)
    
    case words do
      [first | rest] ->
        fixed_first = convert_gerund_to_base(first)
        Enum.join([fixed_first | rest], " ")
      _ -> text
    end
  end

  defp convert_gerund_to_base(word) do
    if String.ends_with?(word, "ing") and String.length(word) > 4 do
      # Remove "ing" suffix
      base = String.slice(word, 0, String.length(word) - 3)
      
      cond do
        # Double consonant: "running" -> "runn" -> "run"
        String.length(base) > 2 and
        String.at(base, -1) == String.at(base, -2) and
        String.at(base, -1) in ~w(n t p m g b d) ->
          String.slice(base, 0, String.length(base) - 1)
        
        # Ends in consonant, might need 'e': "visiting" -> "visit" (already correct)
        # "making" -> "mak" -> "make"
        String.at(base, -1) in ~w(k v c z) ->
          base <> "e"
        
        # Default: just use base
        true ->
          base
      end
    else
      word
    end
  end

  defp format_participants([]), do: "They"
  defp format_participants([single]), do: String.capitalize(single)
  defp format_participants([p1, p2]), do: "#{String.capitalize(p1)} and #{String.capitalize(p2)}"
  defp format_participants(participants) do
    [last | rest] = Enum.reverse(participants)
    formatted = rest |> Enum.reverse() |> Enum.map(&String.capitalize/1) |> Enum.join(", ")
    "#{formatted} and #{String.capitalize(last)}"
  end

  defp clean_sentence(nil), do: nil
  defp clean_sentence(sentence) do
    sentence
    |> String.trim()
    |> ensure_capitalization()
    |> ensure_punctuation()
    |> fix_spacing()
  end

  defp ensure_capitalization(""), do: ""
  defp ensure_capitalization(sentence) do
    first_char = String.at(sentence, 0)
    rest = String.slice(sentence, 1..-1//1)

    String.upcase(first_char) <> rest
  end

  defp ensure_punctuation(""), do: ""
  defp ensure_punctuation(sentence) do
    if String.ends_with?(sentence, ".") or
       String.ends_with?(sentence, "!") or
       String.ends_with?(sentence, "?") do
      sentence
    else
      sentence <> "."
    end
  end

  defp fix_spacing(sentence) do
    sentence
    |> String.replace(~r/\s+/, " ")
    |> String.replace(~r/\s+([.,!?])/, "\\1")
    |> String.replace(~r/\.+/, ".")
    # Fix contraction spacing: "won' t" -> "won't"
    |> String.replace(~r/(\w)\s*'\s*(\w)/, "\\1'\\2")
    # Fix spacing around parentheses
    |> String.replace(~r/\(\s+/, "(")
    |> String.replace(~r/\s+\)/, ")")
    # Remove angle brackets from file tokens
    |> String.replace(~r/<\s*file_\w+\s*>/, "")
  end
end
