defmodule ChatBot.Response.Synthesizer do
  @moduledoc """
  Principle-driven response composition for epistemic queries.

  Instead of template-based responses, this module composes responses
  from primitives based on:
  - Confidence levels of the knowledge being shared
  - Disclosure policy decisions
  - User's interaction style
  - Social appropriateness

  Response primitives:
  - :soft_preface - "From what I remember..."
  - :confidence_bounded - Adjust language to confidence level
  - :evidence_clause - "...based on what you've shared..."
  - :uncertainty_marker - "...but that's just my understanding"
  - :invite_correction - "Feel free to correct me if I'm wrong"

  This enables novel, appropriate responses without massive training data.
  """

  alias ChatBot.Epistemic.Types.SelfKnowledgeAssessment
  alias ChatBot.Epistemic.DisclosurePolicy

  require Logger

  # Response primitives
  @soft_prefaces [
    "From what I remember",
    "Based on our conversations",
    "If I recall correctly",
    "From what you've shared"
  ]

  @evidence_clauses [
    "based on what you've mentioned",
    "from our previous chats",
    "according to what you've told me"
  ]

  @uncertainty_markers [
    "but I could be off",
    "though I might be misremembering",
    "but that's just my impression",
    "though I'm not entirely certain"
  ]

  @correction_invites [
    "Feel free to correct me if I'm wrong.",
    "Let me know if I've got anything mixed up.",
    "Please tell me if that's not quite right.",
    "I'm happy to be corrected on any of this."
  ]

  @empty_knowledge_responses [
    "I don't think I have much information about you yet. We're just getting to know each other.",
    "I haven't learned much about you so far. Is there anything you'd like to share?",
    "We haven't really talked much yet, so I don't have much to go on.",
    "I'm still getting to know you. We haven't shared much yet."
  ]

  @doc """
  Synthesizes a response for a meta-cognitive query.

  Takes a SelfKnowledgeAssessment and produces a natural, appropriate
  response with proper hedging.
  """
  def synthesize_self_knowledge_response(%SelfKnowledgeAssessment{} = assessment, opts \\ []) do
    context = Keyword.get(opts, :context, %{})
    rhetorical_strategy = determine_rhetorical_strategy(assessment, context)

    # Filter through disclosure policy
    filtered = DisclosurePolicy.filter_discloseable(assessment, context)

    # Build response based on what we can disclose
    case rhetorical_strategy do
      :no_knowledge ->
        synthesize_no_knowledge_response(opts)

      :limited_knowledge ->
        synthesize_limited_knowledge_response(filtered, opts)

      :moderate_knowledge ->
        synthesize_moderate_knowledge_response(filtered, opts)

      :rich_knowledge ->
        synthesize_rich_knowledge_response(filtered, opts)
    end
  end

  @doc """
  Synthesizes a response for a single fact with appropriate hedging.
  """
  def synthesize_fact_mention(fact, hedging_level, _opts \\ []) do
    key_human = humanize_key(fact.key)
    value = format_value(fact.value)

    case hedging_level do
      :none ->
        "#{key_human}: #{value}"

      :light ->
        "I think #{key_human} is #{value}"

      :strong ->
        "I might be wrong, but I believe #{key_human} is #{value}"

      :do_not_disclose ->
        nil
    end
  end

  @doc """
  Determines the rhetorical strategy based on assessment content.
  """
  def determine_rhetorical_strategy(%SelfKnowledgeAssessment{} = assessment, _context) do
    discloseable_count = length(assessment.discloseable)
    uncertain_count = length(assessment.inferred_uncertain)
    total = discloseable_count + uncertain_count

    cond do
      total == 0 -> :no_knowledge
      total <= 2 -> :limited_knowledge
      total <= 5 -> :moderate_knowledge
      true -> :rich_knowledge
    end
  end

  @doc """
  Gets response primitives for building custom responses.
  """
  def get_primitives do
    %{
      soft_prefaces: @soft_prefaces,
      evidence_clauses: @evidence_clauses,
      uncertainty_markers: @uncertainty_markers,
      correction_invites: @correction_invites
    }
  end

  # ============================================================================
  # Private Synthesis Functions
  # ============================================================================

  defp synthesize_no_knowledge_response(_opts) do
    Enum.random(@empty_knowledge_responses)
  end

  defp synthesize_limited_knowledge_response(assessment, opts) do
    facts = assessment.discloseable ++ assessment.inferred_uncertain

    if length(facts) == 0 do
      synthesize_no_knowledge_response(opts)
    else
      preface = Enum.random(@soft_prefaces)
      fact_mentions = build_fact_mentions(facts, 2)

      uncertainty =
        if length(assessment.inferred_uncertain) > 0 do
          ", " <> Enum.random(@uncertainty_markers)
        else
          ""
        end

      correction = Enum.random(@correction_invites)

      "#{preface}, #{fact_mentions}#{uncertainty}. #{correction}"
    end
  end

  defp synthesize_moderate_knowledge_response(assessment, opts) do
    high_conf = assessment.discloseable
    uncertain = assessment.inferred_uncertain

    if length(high_conf) == 0 and length(uncertain) == 0 do
      synthesize_no_knowledge_response(opts)
    else
      # Start with preface
      preface = Enum.random(@soft_prefaces)

      # Add high confidence facts
      high_conf_part =
        if length(high_conf) > 0 do
          high_conf_text = build_fact_mentions(high_conf, 3)
          "you've mentioned #{high_conf_text}"
        else
          nil
        end

      # Add uncertain facts with hedging
      uncertain_part =
        if length(uncertain) > 0 do
          uncertain_text = build_fact_mentions(uncertain, 2)
          "I also got the impression that #{uncertain_text}, but I'm not certain about that"
        else
          nil
        end

      # Add evidence clause
      evidence = Enum.random(@evidence_clauses)

      # Build parts list
      parts =
        [preface, high_conf_part, uncertain_part, evidence]
        |> Enum.filter(&(&1 != nil))

      # Build and add closing
      closing = Enum.random(@correction_invites)

      build_flowing_response(parts, closing)
    end
  end

  defp synthesize_rich_knowledge_response(assessment, opts) do
    high_conf = Enum.take(assessment.discloseable, 4)
    uncertain = Enum.take(assessment.inferred_uncertain, 2)

    if length(high_conf) == 0 and length(uncertain) == 0 do
      synthesize_no_knowledge_response(opts)
    else
      # For rich knowledge, we structure more carefully
      intro = "#{Enum.random(@soft_prefaces)}, here's what I know:"

      # Group facts by theme if possible
      high_conf_text =
        if length(high_conf) > 0 do
          build_fact_list(high_conf)
        else
          ""
        end

      uncertain_text =
        if length(uncertain) > 0 do
          "I'm less sure about: #{build_fact_mentions(uncertain, 3)}"
        else
          ""
        end

      # Epistemic disclaimer
      disclaimer =
        "That said, this is all #{Enum.random(@evidence_clauses)}, " <>
          "so it's pretty limited. #{Enum.random(@correction_invites)}"

      [intro, high_conf_text, uncertain_text, disclaimer]
      |> Enum.filter(&(&1 != ""))
      |> Enum.join(" ")
    end
  end

  defp build_fact_mentions(facts, max_count) do
    facts
    |> Enum.take(max_count)
    |> Enum.map(&format_single_fact/1)
    |> join_with_and()
  end

  defp build_fact_list(facts) do
    facts
    |> Enum.map(&format_single_fact/1)
    |> Enum.map(&("- " <> &1))
    |> Enum.join("; ")
  end

  defp format_single_fact(fact) do
    key_human = humanize_key(fact.key)
    value = format_value(fact.value)

    # Vary the phrasing
    templates = [
      "#{key_human} is #{value}",
      "your #{key_human} (#{value})",
      "you mentioned #{key_human}: #{value}",
      "#{value} (#{key_human})"
    ]

    Enum.random(templates)
  end

  defp humanize_key(key) when is_atom(key) do
    key
    |> Atom.to_string()
    |> humanize_key()
  end

  defp humanize_key(key) when is_binary(key) do
    key
    |> String.replace("_", " ")
    |> String.replace("-", " ")
  end

  defp humanize_key(_), do: "something"

  defp format_value(value) when is_binary(value), do: value
  defp format_value(value) when is_atom(value), do: Atom.to_string(value)
  defp format_value(value) when is_number(value), do: to_string(value)
  defp format_value(value) when is_list(value), do: Enum.join(value, ", ")
  defp format_value(value), do: inspect(value)

  defp join_with_and([]), do: ""
  defp join_with_and([single]), do: single

  defp join_with_and([a, b]), do: "#{a} and #{b}"

  defp join_with_and(items) do
    {last, rest} = List.pop_at(items, -1)
    Enum.join(rest, ", ") <> ", and #{last}"
  end

  defp build_flowing_response(parts, closing) do
    # Join parts into a flowing sentence
    main =
      parts
      |> Enum.filter(&(&1 != nil and &1 != ""))
      |> Enum.join(", ")

    "#{main}. #{closing}"
  end
end
