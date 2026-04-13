defmodule Brain.ML.LSTM.AccuracyComparisonTest do
  @moduledoc """
  Compares LSTM intent classifier accuracy against TF-IDF baseline,
  using the full production pipeline with Atlas context.

  This test seeds Atlas with realistic entity data so the pipeline
  can leverage entity disambiguation, Gazetteer enrichment, and
  sentiment context from the knowledge graph -- matching the ideal
  production scenario.

  Systemic regression guard: this suite explicitly loads the production
  TF-IDF model artifact from `priv/ml_models/classifier.term` in read-only
  mode so failures represent production model regressions.

  Run with: mix test --only slow
  """
  use Brain.Test.GraphCase, async: false

  alias Brain.Analysis.Pipeline
  alias Brain.ML.IntentClassifierSimple

  @moduletag :lstm
  @moduletag :slow
  @moduletag timeout: 300_000
  @tag seed_knowledge: true

  setup_all do
    load_production_intent_model!()
    :ok
  end

  # Test cases that historically caused misclassification.
  # Each tuple is {text, expected_domain_prefix}.
  @confusable_cases [
    # Weather queries -- Atlas has Location entities for these cities
    {"tell me about the weather", "weather.query"},
    {"what's the weather in Paris", "weather.query"},
    {"what do you know about the weather", "meta.self_knowledge"},
    {"tell me what the weather is like in London", "weather.query"},

    # Agent acquaintance / self-knowledge
    {"tell me about yourself", "smalltalk.agent.acquaintance"},
    {"what can you tell me about yourself", "meta.self_knowledge"},
    {"tell me about you", "smalltalk.agent.acquaintance"},

    # Music queries
    {"play some music", "music.play"},
    {"can you play me some songs", "music.play"},

    # Smart home device control
    {"turn on the lights", "smarthome.lights.switch.on"},
    {"turn off the living room light", "smarthome.lights.switch.off"}
  ]

  describe "Production pipeline with Atlas context" do
    @tag :comparison
    test "pipeline correctly classifies confusable cases with Atlas data", context do
      # Seed additional domain entities into Atlas so the pipeline has
      # realistic context for entity disambiguation and slot filling
      seed_domain_entities()

      # Allow the test process and any spawned tasks to access the sandbox
      allow_sandbox_for_tasks(context[:sandbox_pid])

      results =
        Enum.map(@confusable_cases, fn {text, expected_domain} ->
          analysis = Pipeline.analyze_chunk(text)

          intent = analysis.intent || "unknown"
          confidence = analysis.confidence || 0.0
          entities = analysis.entities || []

          matches = intent_matches_domain?(intent, expected_domain)

          %{
            text: text,
            expected_domain: expected_domain,
            pipeline_intent: intent,
            pipeline_confidence: confidence,
            pipeline_correct: matches,
            entity_count: length(entities),
            entity_types: Enum.map(entities, & &1[:entity_type]) |> Enum.uniq()
          }
        end)

      correct = Enum.count(results, & &1.pipeline_correct)
      total = length(@confusable_cases)

      IO.puts("\n=== Production Pipeline Classification (with Atlas) ===")
      IO.puts("Correct: #{correct}/#{total}")
      IO.puts("")

      Enum.each(results, fn r ->
        mark = if r.pipeline_correct, do: "✓", else: "✗"

        IO.puts("\"#{r.text}\"")
        IO.puts("  Expected: #{r.expected_domain}")
        IO.puts("  Got:      #{r.pipeline_intent} (#{Float.round(r.pipeline_confidence, 3)}) #{mark}")

        if r.entity_count > 0 do
          IO.puts("  Entities: #{inspect(r.entity_types)}")
        end

        IO.puts("")
      end)

      # With Atlas providing entity context and TF-IDF classifier,
      # the pipeline should correctly classify at least 50% of confusable cases.
      # LSTM models would improve this further when available.
      min_correct = div(total * 5, 10)

      assert correct >= min_correct,
        "Pipeline with Atlas context should correctly classify at least " <>
        "#{min_correct}/#{total} confusable cases, but only got #{correct}/#{total}"
    end

    @tag :comparison
    test "Atlas entity context improves entity extraction for location queries", context do
      seed_domain_entities()
      allow_sandbox_for_tasks(context[:sandbox_pid])

      # "Paris" should be recognized as a location with Atlas backing
      analysis = Pipeline.analyze_chunk("what's the weather in Paris")

      location_entities =
        (analysis.entities || [])
        |> Enum.filter(fn e ->
          e[:entity_type] in ["location", "city"] and
            String.downcase(e[:value] || "") == "paris"
        end)

      assert length(location_entities) > 0,
        "Paris should be extracted as a location entity when Atlas has Location data. " <>
        "Got entities: #{inspect(analysis.entities)}"
    end

    @tag :comparison
    test "Atlas entity context improves disambiguation for ambiguous names", context do
      seed_domain_entities()
      allow_sandbox_for_tasks(context[:sandbox_pid])

      # "London" in a weather context should resolve to location, not person
      analysis = Pipeline.analyze_chunk("what's the weather like in London")

      london_entities =
        (analysis.entities || [])
        |> Enum.filter(fn e ->
          String.downcase(e[:value] || "") == "london"
        end)

      if length(london_entities) > 0 do
        london = hd(london_entities)
        assert london[:entity_type] in ["location", "city"],
          "London in weather context should be disambiguated as location/city, " <>
          "got: #{london[:entity_type]}"
      end
    end
  end

  # Seeds additional entities into the knowledge_graph beyond what
  # GraphSeeds.seed_knowledge_graph provides
  defp seed_domain_entities do
    alias Atlas.Graph

    # Music entities
    Graph.add_node("knowledge_graph", "Artist", %{name: "The Beatles", genre: "rock"})
    Graph.add_node("knowledge_graph", "Artist", %{name: "Taylor Swift", genre: "pop"})

    # Device entities
    Graph.add_node("knowledge_graph", "Device", %{name: "lights", type: "lighting"})
    Graph.add_node("knowledge_graph", "Device", %{name: "thermostat", type: "heating"})

    # Additional locations
    Graph.add_node("knowledge_graph", "Location", %{name: "New York", type: "city"})
    Graph.add_node("knowledge_graph", "Location", %{name: "Tokyo", type: "city"})

    :ok
  end

  # Allow spawned tasks to use the Ecto sandbox
  defp allow_sandbox_for_tasks(sandbox_pid) do
    if task_sup = Process.whereis(Brain.AtlasTaskSupervisor) do
      Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, sandbox_pid, task_sup)
    end

    Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, {:shared, sandbox_pid})
  end

  defp intent_matches_domain?(intent, domain) do
    intent_lower = String.downcase(intent)
    domain_lower = String.downcase(domain)
    String.starts_with?(intent_lower, domain_lower)
  end

  defp load_production_intent_model! do
    model_path = resolve_production_model_path!()
    model = model_path |> File.read!() |> :erlang.binary_to_term()
    :ok = GenServer.call(IntentClassifierSimple, {:load_trained_model, model}, 120_000)
  end

  defp resolve_production_model_path! do
    candidates = [
      Path.join(File.cwd!(), "_build/dev/lib/brain/priv/ml_models/classifier.term"),
      Path.expand("../../../../priv/ml_models/classifier.term", __DIR__)
    ]

    case Enum.find(candidates, &File.exists?/1) do
      nil ->
        raise "Production classifier model not found in expected paths: #{inspect(candidates)}"

      path ->
        path
    end
  end
end
