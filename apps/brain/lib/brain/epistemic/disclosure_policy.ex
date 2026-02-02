defmodule Brain.Epistemic.DisclosurePolicy do
  @moduledoc """
  Governs what information is appropriate to share with users.

  Implements the validation questions from epistemic framing research:
  - V6: Is this information socially appropriate to disclose right now?
  - V7: Would this sound creepy if said confidently?
  - V8: Should I hedge, ask permission, or generalize?

  The policy evaluates each piece of knowledge and returns a
  DisclosureDecision indicating:
  - Whether to disclose at all
  - What level of hedging is required
  - Whether to generalize or be specific
  - Whether to ask permission first
  """

  alias Brain.Epistemic.Types.{Belief, DisclosureDecision, SelfKnowledgeAssessment}

  require Logger

  # Predicates that are always safe to disclose
  @always_safe_predicates [
    :name,
    :preferred_name,
    :timezone,
    :language,
    :greeting_preference
  ]

  # Predicates that require high confidence to disclose
  @require_high_confidence [
    :occupation,
    :workplace,
    :relationship_status,
    :age,
    :location
  ]

  # Predicates that should never be disclosed (even if known)
  @never_disclose [
    :password,
    :ssn,
    :social_security,
    :credit_card,
    :bank_account,
    :health_condition,
    :medical_history
  ]

  # Predicates that might sound creepy if stated too confidently
  @creepy_if_confident [
    :daily_routine,
    :sleep_schedule,
    :location_history,
    :browsing_history,
    :emotional_state,
    :personality_traits,
    :relationships
  ]

  @doc """
  Evaluates whether a belief should be disclosed.

  Context options:
  - :relationship_duration - How long we've known the user (:new | :established | :long_term)
  - :conversation_tone - Current tone (:formal | :casual | :technical)
  - :user_initiated - Whether user asked about this specifically
  - :previous_disclosures - What we've disclosed before
  """
  def evaluate_disclosure(%Belief{} = belief, context \\ %{}) do
    predicate = normalize_predicate(belief.predicate)

    # Run through validation checks
    cond do
      # V1: Never disclose sensitive information
      predicate in @never_disclose ->
        DisclosureDecision.do_not_disclose("Sensitive information - never disclose")

      # V2: Check if predicate is always safe
      predicate in @always_safe_predicates and belief.source == :explicit ->
        DisclosureDecision.disclose("Safe predicate with explicit source")

      # V3: Low confidence requires strong hedging or no disclosure
      belief.confidence < 0.3 ->
        DisclosureDecision.do_not_disclose("Confidence too low (#{belief.confidence})")

      # V4: Inferred knowledge about personal matters needs permission
      belief.source == :inferred and predicate in @require_high_confidence ->
        DisclosureDecision.ask_first("Inferred personal information - seek permission")

      # V5: Check for creepiness
      would_be_creepy?(belief, context) ->
        if belief.confidence >= 0.7 do
          DisclosureDecision.disclose_with_hedging(:strong, "Could sound creepy if too confident")
        else
          DisclosureDecision.do_not_disclose("Too uncertain and potentially creepy")
        end

      # V6: Moderate confidence needs hedging
      belief.confidence < 0.6 ->
        DisclosureDecision.disclose_with_hedging(:strong, "Moderate confidence")

      belief.confidence < 0.8 ->
        DisclosureDecision.disclose_with_hedging(:light, "Good confidence but not certain")

      # V7: High confidence explicit can be stated directly
      belief.source == :explicit and belief.confidence >= 0.8 ->
        DisclosureDecision.disclose("High confidence explicit statement")

      # V8: High confidence inferred still needs light hedging
      belief.source == :inferred and belief.confidence >= 0.8 ->
        DisclosureDecision.disclose_with_hedging(:light, "High confidence but inferred")

      # Default: light hedging
      true ->
        DisclosureDecision.disclose_with_hedging(:light, "Default policy")
    end
  end

  @doc """
  Evaluates disclosure for an entire SelfKnowledgeAssessment.

  Returns a map of fact_key => DisclosureDecision.
  """
  def evaluate_assessment(%SelfKnowledgeAssessment{} = assessment, context \\ %{}) do
    all_facts =
      assessment.discloseable ++
        assessment.inferred_uncertain ++
        assessment.should_avoid

    all_facts
    |> Enum.map(fn fact ->
      belief =
        Belief.new(:user, fact.key, fact.value,
          confidence: fact.confidence,
          source: fact.provenance
        )

      {fact.key, evaluate_disclosure(belief, context)}
    end)
    |> Map.new()
  end

  @doc """
  Filters an assessment to only include facts that should be disclosed.

  Returns a new assessment with filtered facts.
  """
  def filter_discloseable(%SelfKnowledgeAssessment{} = assessment, context \\ %{}) do
    decisions = evaluate_assessment(assessment, context)

    filter_by_decisions = fn facts ->
      Enum.filter(facts, fn fact ->
        case Map.get(decisions, fact.key) do
          %DisclosureDecision{should_disclose: true} -> true
          _ -> false
        end
      end)
    end

    %{
      assessment
      | discloseable: filter_by_decisions.(assessment.discloseable),
        inferred_uncertain: filter_by_decisions.(assessment.inferred_uncertain),
        should_avoid: []
    }
  end

  @doc """
  Gets the hedging level for a fact.

  Returns :none | :light | :strong | :do_not_disclose
  """
  def get_hedging_level(fact, context \\ %{}) do
    belief =
      Belief.new(:user, fact.key, fact.value,
        confidence: fact.confidence,
        source: fact.provenance
      )

    decision = evaluate_disclosure(belief, context)

    if decision.should_disclose do
      decision.hedging_required
    else
      :do_not_disclose
    end
  end

  @doc """
  Checks if disclosing this belief would violate any policy.
  """
  def violates_policy?(%Belief{} = belief, context \\ %{}) do
    decision = evaluate_disclosure(belief, context)
    not decision.should_disclose
  end

  @doc """
  Gets all predicates that are safe to disclose without hedging.
  """
  def safe_predicates, do: @always_safe_predicates

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp normalize_predicate(predicate) when is_atom(predicate), do: predicate

  defp normalize_predicate(predicate) when is_binary(predicate) do
    predicate
    |> String.downcase()
    |> String.replace([" ", "-"], "_")
    |> String.to_atom()
  end

  defp normalize_predicate(_), do: :unknown

  defp would_be_creepy?(%Belief{} = belief, context) do
    predicate = normalize_predicate(belief.predicate)
    relationship = Map.get(context, :relationship_duration, :new)
    user_initiated = Map.get(context, :user_initiated, false)

    cond do
      # User asked specifically - less creepy
      user_initiated ->
        false

      # Creepy predicates with new relationship
      predicate in @creepy_if_confident and relationship == :new ->
        true

      # Inferred personal details early in relationship
      belief.source == :inferred and relationship == :new and
          predicate in @require_high_confidence ->
        true

      # Location tracking vibes
      predicate in [:location, :location_history, :daily_routine] and belief.source != :explicit ->
        true

      true ->
        false
    end
  end
end
