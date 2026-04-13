defmodule Brain.Epistemic.DisclosurePolicyTest do
  alias Brain.Epistemic.Types
  use ExUnit.Case, async: false

  alias Brain.Epistemic.DisclosurePolicy
  alias Types.{Belief, SelfKnowledgeAssessment}

  describe "evaluate_disclosure/2" do
    test "allows disclosure of safe explicit facts" do
      belief =
        Belief.new(:user, :name, "Alice", confidence: 0.95, source: :explicit)

      decision = DisclosurePolicy.evaluate_disclosure(belief)

      assert decision.should_disclose == true
      assert decision.hedging_required == :none
    end

    test "blocks disclosure of sensitive information" do
      belief =
        Belief.new(:user, :password, "secret123", confidence: 1.0, source: :explicit)

      decision = DisclosurePolicy.evaluate_disclosure(belief)

      assert decision.should_disclose == false
      assert decision.reason =~ "Sensitive"
    end

    test "blocks disclosure of very low confidence facts" do
      belief =
        Belief.new(:user, :hobby, "skydiving", confidence: 0.2, source: :inferred)

      decision = DisclosurePolicy.evaluate_disclosure(belief)

      assert decision.should_disclose == false
      assert decision.reason =~ "Confidence"
    end

    test "requires hedging for moderate confidence" do
      belief =
        Belief.new(:user, :hobby, "hiking", confidence: 0.55, source: :inferred)

      decision = DisclosurePolicy.evaluate_disclosure(belief)

      assert decision.should_disclose == true
      assert decision.hedging_required == :strong
    end

    test "requires light hedging for good confidence" do
      belief =
        Belief.new(:user, :timezone, "PST", confidence: 0.75, source: :inferred)

      decision = DisclosurePolicy.evaluate_disclosure(belief)

      assert decision.should_disclose == true
      assert decision.hedging_required == :light
    end

    test "requires permission for inferred personal information" do
      belief =
        Belief.new(:user, :occupation, "Doctor", confidence: 0.7, source: :inferred)

      decision = DisclosurePolicy.evaluate_disclosure(belief)

      assert decision.ask_permission == true or decision.hedging_required != :none
    end

    test "adds hedging for potentially creepy information" do
      belief =
        Belief.new(:user, :daily_routine, "Wakes at 6am", confidence: 0.9, source: :inferred)

      context = %{relationship_duration: :new}
      decision = DisclosurePolicy.evaluate_disclosure(belief, context)

      assert decision.hedging_required == :strong or decision.should_disclose == false
    end

    test "less creepy when user initiated" do
      belief =
        Belief.new(:user, :timezone, "PST", confidence: 0.85, source: :inferred)

      context = %{user_initiated: true}
      decision = DisclosurePolicy.evaluate_disclosure(belief, context)

      assert decision.should_disclose == true
    end
  end

  describe "evaluate_assessment/2" do
    test "evaluates all facts in assessment" do
      assessment = %SelfKnowledgeAssessment{
        user_id: "user1",
        discloseable: [
          %{key: :name, value: "Alice", confidence: 0.95, provenance: :explicit},
          %{key: :location, value: "Seattle", confidence: 0.85, provenance: :explicit}
        ],
        inferred_uncertain: [
          %{key: :occupation, value: "Engineer", confidence: 0.6, provenance: :inferred}
        ],
        should_avoid: [],
        total_facts: 3,
        assessment_timestamp: DateTime.utc_now()
      }

      decisions = DisclosurePolicy.evaluate_assessment(assessment)

      assert is_map(decisions)
      assert Map.has_key?(decisions, :name)
      assert Map.has_key?(decisions, :location)
      assert Map.has_key?(decisions, :occupation)
    end
  end

  describe "filter_discloseable/2" do
    test "filters out facts that should not be disclosed" do
      assessment = %SelfKnowledgeAssessment{
        user_id: "user1",
        discloseable: [
          %{key: :name, value: "Alice", confidence: 0.95, provenance: :explicit},
          %{key: :password, value: "secret", confidence: 1.0, provenance: :explicit}
        ],
        inferred_uncertain: [
          %{key: :occupation, value: "Engineer", confidence: 0.6, provenance: :inferred}
        ],
        should_avoid: [],
        total_facts: 3,
        assessment_timestamp: DateTime.utc_now()
      }

      filtered = DisclosurePolicy.filter_discloseable(assessment)
      discloseable_keys = Enum.map(filtered.discloseable, & &1.key)
      refute :password in discloseable_keys
      assert :name in discloseable_keys
    end
  end

  describe "get_hedging_level/2" do
    test "returns correct hedging level" do
      high_conf_fact = %{key: :name, value: "Alice", confidence: 0.95, provenance: :explicit}
      med_conf_fact = %{key: :hobby, value: "Coding", confidence: 0.65, provenance: :inferred}
      low_conf_fact = %{key: :age, value: "30", confidence: 0.25, provenance: :inferred}

      assert DisclosurePolicy.get_hedging_level(high_conf_fact) == :none
      assert DisclosurePolicy.get_hedging_level(med_conf_fact) == :light
      assert DisclosurePolicy.get_hedging_level(low_conf_fact) == :do_not_disclose
    end
  end

  describe "violates_policy?/2" do
    test "returns true for sensitive data" do
      belief =
        Belief.new(:user, :ssn, "123-45-6789", confidence: 1.0, source: :explicit)

      assert DisclosurePolicy.violates_policy?(belief) == true
    end

    test "returns false for safe data" do
      belief =
        Belief.new(:user, :name, "Bob", confidence: 0.95, source: :explicit)

      assert DisclosurePolicy.violates_policy?(belief) == false
    end
  end

  describe "safe_predicates/0" do
    test "returns list of safe predicates" do
      safe = DisclosurePolicy.safe_predicates()

      assert is_list(safe)
      assert :name in safe
      assert :timezone in safe
    end
  end
end