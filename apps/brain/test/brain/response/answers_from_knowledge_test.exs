defmodule Brain.Response.AnswersFromKnowledgeTest do
  @moduledoc """
  A chunk that asks is answered from what is known about it, on the real path
  a turn takes: `Brain.Analysis.Pipeline.process/2` retrieves the facts about
  what the question names and aligns a polar question's proposition with them,
  `Brain.Response.ContextBuilder` carries the known statements,
  `Brain.Response.DiscoursePlanner` plans from `has_facts` and
  `fact_alignment`, `Brain.Response.ContentSpecifier` fills the plan with the
  statements, and `Brain.Response.SurfaceRealizer` words it.

  The curated facts of the test fixtures (`config :brain, :facts_dir`) are
  imported into the test's sandbox and the FactDatabase reloaded from them;
  when the test ends they are deleted and the FactDatabase reloaded again, back
  to the empty database it boots from.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Analysis.{ChunkPriority, Pipeline}
  alias Brain.FactDatabase
  alias Brain.Response.{ContentSpecifier, ContextBuilder, DiscoursePlanner, SurfaceRealizer}

  @world_id "default"

  setup do
    {:ok, %{"learned_facts" => {:ok, imported}}} = Atlas.Importer.import_all(only: ["learned_facts"], quiet: true)
    assert imported > 0
    :ok = FactDatabase.reload()

    on_exit(fn ->
      Atlas.Repo.delete_all(Atlas.Schemas.LearnedFact)
      :ok = FactDatabase.reload()
    end)

    :ok
  end

  # The turn as the refinement loop plans, specifies and realizes it.
  defp respond(text) do
    opts = [
      world_id: @world_id,
      side_effects: false,
      time_zone: Brain.Temporal.Zone.utc(),
      turn_time: DateTime.utc_now(),
      participants: [:user, :bot],
      conversation_history: [],
      user_profile: %{}
    ]

    model = Pipeline.process(text, opts)
    loop_opts = Keyword.put(opts, :unified_context, ContextBuilder.build(model, opts))
    plan = DiscoursePlanner.plan(model, loop_opts)
    primary = ChunkPriority.select_primary(model.analyses)
    specified = ContentSpecifier.specify(plan, primary, loop_opts)

    {:ok, _rendered, reply, _pattern_use} =
      SurfaceRealizer.realize(
        specified,
        Keyword.merge(loop_opts,
          analysis: primary,
          analyses: model.analyses,
          feature_vector: primary.feature_vector,
          intent: primary.intent
        )
      )

    %{analysis: primary, plan: Enum.map(plan, &{&1.type, &1.variant}), reply: reply}
  end

  @capital "The capital of France is Paris"

  test "a polar question the facts confirm is answered yes, with the fact that confirms it" do
    %{analysis: analysis, plan: plan, reply: reply} = respond("Is Paris the capital of France?")

    assert analysis.fact_alignment == %{alignment: :confirming, statement: @capital}
    assert plan == [{:framing, :affirmative}, {:content, :factual}]
    assert reply in ["Yes, the capital of France is Paris.", "That's right, the capital of France is Paris."]
  end

  test "a polar question a negated fact disconfirms is answered no, with that fact" do
    {:ok, _id} =
      FactDatabase.add_fact_direct(%{
        "id" => "test_not_capital",
        "entity" => "Germany",
        "fact" => "Paris is not the capital of Germany",
        "category" => "geography"
      })

    %{analysis: analysis, plan: plan, reply: reply} = respond("Is Paris the capital of Germany?")

    assert analysis.fact_alignment == %{alignment: :disconfirming, statement: "Paris is not the capital of Germany"}
    assert plan == [{:framing, :negative}, {:content, :factual}]
    assert reply in ["No, Paris is not the capital of Germany.", "No, that's not right. Paris is not the capital of Germany."]
  end

  test "a polar question no fact states is never answered yes or no, only with what is known" do
    %{analysis: analysis, plan: plan, reply: reply} = respond("Is Berlin the capital of France?")

    assert analysis.fact_alignment == nil
    assert plan == [{:framing, :informative}, {:content, :factual}]
    assert reply in ["Here's what I know. #{@capital}.", "Here's what I have on that. #{@capital}."]
  end

  test "a question about what a fact is about is answered with the fact" do
    %{plan: plan, reply: reply} = respond("What is the capital of France?")

    assert plan == [{:framing, :informative}, {:content, :factual}]
    assert reply in ["Here's what I know. #{@capital}.", "Here's what I have on that. #{@capital}."]
  end

  test "an explanation asked for is given from the fact about what it names" do
    %{plan: plan, reply: reply} = respond("Explain how photosynthesis works")

    photosynthesis =
      "Photosynthesis is the process by which plants convert carbon dioxide and water into glucose using sunlight energy"

    assert plan == [{:framing, :informative}, {:content, :explanatory}]
    assert reply in ["Here's what I know. #{photosynthesis}.", "Here's what I have on that. #{photosynthesis}."]
  end

  test "a question that names no fact's entity is answered from no fact" do
    # "time zones" is a fact's entity, and "time" a word of it: only an entity
    # named whole is what a question is about.
    %{analysis: analysis, reply: reply} = respond("What time is it?")

    assert analysis.related_facts == []
    refute reply =~ "time zones"
  end
end
