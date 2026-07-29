defmodule Brain.Response.ContentSpecifierTest do
  @moduledoc """
  Guards the belief path into a narrative primitive — the point where stored
  beliefs become prompt content.

  Both behaviours here were broken on main, and each masked the other:

    * `retrieve_beliefs/1` matched `beliefs when is_list(beliefs)` against
      `query_beliefs/1`'s `{:ok, beliefs}` reply, so it returned `[]` every time
      the store was actually up — no belief ever reached a primitive.
    * `apply_disclosure_filter/1` called `DisclosurePolicy.filter_discloseable/1`
      with a list, which only clauses on `%SelfKnowledgeAssessment{}`. The
      resulting FunctionClauseError was swallowed by a blanket `rescue -> beliefs`,
      so the filter was a no-op.

  Because retrieval returned nothing, the inert filter was invisible. Fixing
  retrieval without the filter would have sent `:ssn` and `:health_condition` to
  the model, so both are asserted together.
  """
  alias Brain.Epistemic
  use Brain.Test.GraphCase, async: false

  alias Epistemic.{BeliefStore, JTMS}
  alias Brain.Response.{ContentSpecifier, Primitive}
  alias Brain.Analysis.ChunkAnalysis
  import Brain.TestHelpers

  setup do
    start_brain_services()
    ensure_started(BeliefStore)
    ensure_started(JTMS)
    BeliefStore.clear()
    :ok
  end

  defp narrative_beliefs do
    analysis = %ChunkAnalysis{text: "tell me about myself", entities: [], confidence: 0.8}
    primitive = %Primitive{type: :content, variant: :narrative, content: %{}}
    [specified] = ContentSpecifier.specify([primitive], analysis, [])
    specified
  end

  describe "beliefs reaching a narrative primitive" do
    test "a safe, high-confidence belief actually flows through" do
      {:ok, _} =
        BeliefStore.add_belief(:user, :name, "Austin", confidence: 0.95, source: :explicit)

      specified = narrative_beliefs()
      kept = specified.content[:beliefs] || []

      assert kept != [], "no belief reached the primitive — retrieve_beliefs/1 returned []"
      assert :name in Enum.map(kept, & &1.predicate)
      assert specified.content[:belief_count] == length(kept)
    end

    test "@never_disclose predicates are withheld even at high confidence" do
      {:ok, _} =
        BeliefStore.add_belief(:user, :name, "Austin", confidence: 0.95, source: :explicit)

      {:ok, _} =
        BeliefStore.add_belief(:user, :health_condition, "diabetes",
          confidence: 0.99,
          source: :explicit
        )

      {:ok, _} =
        BeliefStore.add_belief(:user, :ssn, "123-45-6789", confidence: 0.99, source: :explicit)

      predicates = narrative_beliefs().content[:beliefs] |> Enum.map(& &1.predicate)

      assert :name in predicates
      refute :health_condition in predicates
      refute :ssn in predicates
    end

    test "beliefs below the 0.3 confidence floor are withheld" do
      {:ok, _} =
        BeliefStore.add_belief(:user, :timezone, "UTC-12", confidence: 0.1, source: :inferred)

      predicates = narrative_beliefs().content[:beliefs] |> Enum.map(& &1.predicate)

      refute :timezone in predicates,
             "a 0.1-confidence belief must not be handed to the model as fact"
    end

    test "confidence flags describe only what survived the filter" do
      {:ok, _} =
        BeliefStore.add_belief(:user, :name, "Austin", confidence: 0.95, source: :explicit)

      {:ok, _} =
        BeliefStore.add_belief(:user, :ssn, "123-45-6789", confidence: 0.99, source: :explicit)

      content = narrative_beliefs().content

      assert content[:has_high_confidence] == true
      assert content[:belief_count] == length(content[:beliefs])
      refute Enum.any?(content[:beliefs], &(&1.predicate == :ssn))
    end
  end
end
