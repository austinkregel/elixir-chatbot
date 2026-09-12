# exercise_brain.exs
#
# Exercises key Brain modules end-to-end, threading outputs forward as inputs.
# Ouro is never called directly -- it gets invoked by the response pipeline
# when the system decides it is the right path.
#
# Run: mix run exercise_brain.exs
# Run with world: mix run exercise_brain.exs -- --world my_world

alias Brain.ML.{MicroClassifiers, EntityExtractor}
alias Brain.Analysis.Pipeline
alias Brain.Memory.Store, as: MemoryStore
alias Brain.Response.{Generator, Enricher}
alias Brain.Services.Dispatcher

# --- CLI arg parsing ---

world_id =
  case System.argv() do
    ["--world", id | _] -> id
    _ -> "default"
  end

# --- Helpers ---

divider = fn title ->
  IO.puts("\n" <> String.duplicate("=", 72))
  IO.puts("  #{title}")
  IO.puts(String.duplicate("=", 72))
end

log_input = fn label, data ->
  IO.puts("  [input] #{label}:")
  IO.inspect(data, label: "         ", pretty: true, limit: 20, width: 100)
end

log_output = fn label, data ->
  IO.puts("  [output] #{label}:")
  IO.inspect(data, label: "          ", pretty: true, limit: 20, width: 100)
end

timed = fn label, fun ->
  start = System.monotonic_time(:millisecond)
  result = fun.()
  elapsed = System.monotonic_time(:millisecond) - start
  IO.puts("  [timing] #{label}: #{elapsed}ms")
  {result, elapsed}
end

section = fn title, acc, fun ->
  divider.(title)

  :erlang.garbage_collect()

  try do
    result = fun.(acc)
    :erlang.garbage_collect()
    result
  rescue
    e ->
      IO.puts("  [ERROR] #{Exception.message(e)}")
      IO.puts("  #{Exception.format(:error, e, __STACKTRACE__) |> String.slice(0, 500)}")
      Map.update(acc, :failed, [title], &[title | &1])
  end
end

log_eval_result = fn label, result ->
  case result do
    {:ok, %{response: response, processing_method: method} = enriched} ->
      IO.puts("  [#{label}] processing_method: #{inspect(method)}")
      IO.puts("  [#{label}] response (#{String.length(response || "")} chars):")
      IO.puts("    #{String.slice(response || "", 0, 300)}")

      if Map.has_key?(enriched, :context) do
        ctx = enriched.context
        IO.puts("  [#{label}] context keys: #{inspect(Map.keys(ctx))}")

        if Map.has_key?(ctx, :intent) do
          IO.puts("  [#{label}] intent: #{inspect(ctx.intent)}")
        end

        if Map.has_key?(ctx, :entities) do
          IO.puts("  [#{label}] entities: #{inspect(ctx.entities, limit: 5)}")
        end
      end

      enriched

    {:ok, other} ->
      IO.puts("  [#{label}] result:")
      IO.inspect(other, label: "    ", pretty: true, limit: 10)
      other

    {:error, reason} ->
      IO.puts("  [#{label}] ERROR: #{inspect(reason)}")
      nil

    other ->
      IO.puts("  [#{label}] unexpected:")
      IO.inspect(other, label: "    ", pretty: true, limit: 10)
      other
  end
end

# --- Startup ---

divider.("Brain Exercise Script")
IO.puts("  World ID: #{world_id}")
IO.puts("  Time:     #{DateTime.utc_now() |> DateTime.to_string()}")

acc = %{
  world_id: world_id,
  intents: [],
  entities: [],
  sentiments: [],
  memories: [],
  pipeline_result: nil,
  enrichment: nil,
  gen_responses: [],
  eval_turns: [],
  failed: []
}

# ============================================================
# 1. Intent Classifier
# ============================================================

acc =
  section.("1. Intent Classifier (MicroClassifiers :intent_full)", acc, fn acc ->
    ready = MicroClassifiers.ready?()
    IO.puts("  MicroClassifiers ready? #{ready}")

    unless ready do
      IO.puts("  [WARN] MicroClassifiers not ready, attempting to continue anyway...")
    end

    samples = [
      {"Hey there, how are you?", "smalltalk.greetings.how_are_you"},
      {"What's the weather like in Denver?", "weather.query"},
      {"Play some jazz music", "music.play"},
      {"Set a reminder for my meeting at 3pm tomorrow", "reminder.create"},
      {"What's the capital of France?", "knowledge.capital"},
      {"Turn off the living room lights", "smarthome.lights.switch.off"},
      {"Goodbye, talk to you later", "smalltalk.greetings.bye"}
    ]

    intents =
      Enum.map(samples, fn {text, expected} ->
        log_input.("classify :intent_full", %{text: text, expected: expected})

        {result, _elapsed} =
          timed.("classify(#{String.slice(text, 0, 30)}...)", fn ->
            MicroClassifiers.classify(:intent_full, text)
          end)

        log_output.("result", result)

        case result do
          {:ok, %{label: label, confidence: confidence}} ->
            %{text: text, intent: label, confidence: confidence, expected: expected}

          {:ok, other} ->
            %{
              text: text,
              intent: Map.get(other, :label, :unknown),
              confidence: Map.get(other, :confidence, 0.0),
              expected: expected
            }

          {:error, reason} ->
            IO.puts("  [WARN] Classification failed: #{inspect(reason)}")
            %{text: text, intent: :unknown, confidence: 0.0, expected: expected}

          other ->
            IO.puts("  [WARN] Unexpected result: #{inspect(other)}")
            %{text: text, intent: :unknown, confidence: 0.0, expected: expected}
        end
      end)

    IO.puts("\n  Accumulated intents:")
    correct = Enum.count(intents, fn i -> to_string(i.intent) == i.expected end)
    total = length(intents)

    Enum.each(intents, fn i ->
      match = if to_string(i.intent) == i.expected, do: "MATCH", else: "MISMATCH"
      IO.puts("    [#{match}] \"#{String.slice(i.text, 0, 45)}\"")

      IO.puts(
        "      actual=#{i.intent} (#{Float.round(i.confidence * 100, 1)}%), expected=#{i.expected}"
      )
    end)

    IO.puts(
      "\n  Intent accuracy: #{correct}/#{total} (#{Float.round(correct / max(total, 1) * 100, 1)}%)"
    )

    %{acc | intents: intents}
  end)

# ============================================================
# 2. Entity Extractor
# ============================================================

acc =
  section.("2. Entity Extractor", acc, fn acc ->
    samples =
      acc.intents
      |> Enum.map(& &1.text)
      |> Enum.concat([
        "Tell me about Elixir programming by Jose Valim",
        "Navigate to 123 Main Street in Portland, Oregon",
        "Send an email to Sarah about the quarterly report"
      ])

    entities =
      Enum.map(samples, fn text ->
        opts = [world_id: acc.world_id]
        log_input.("extract_entities", %{text: text, opts: opts})

        {result, _elapsed} =
          timed.("extract_entities(#{String.slice(text, 0, 30)}...)", fn ->
            EntityExtractor.extract_entities(text, opts)
          end)

        log_output.("entities", result)

        case result do
          entities when is_list(entities) ->
            %{text: text, entities: entities}

          other ->
            IO.puts("  [WARN] Unexpected entity result: #{inspect(other)}")
            %{text: text, entities: []}
        end
      end)

    IO.puts("\n  Accumulated entities:")

    Enum.each(entities, fn e ->
      names =
        Enum.map(e.entities, fn ent ->
          "#{Map.get(ent, :value, Map.get(ent, :text, "?"))}(#{Map.get(ent, :type, "?")})"
        end)

      IO.puts("    - \"#{String.slice(e.text, 0, 40)}\" => [#{Enum.join(names, ", ")}]")
    end)

    %{acc | entities: entities}
  end)

# ============================================================
# 3. Sentiment Classification (TF-IDF)
# ============================================================

acc =
  section.("3. Sentiment Classification", acc, fn acc ->
    sentiment_samples = [
      {"Hey there, how are you?", :neutral},
      {"What's the weather like in Denver?", :neutral},
      {"Play some jazz music", :neutral},
      {"Set a reminder for my meeting at 3pm tomorrow", :neutral},
      {"What's the capital of France?", :neutral},
      {"Turn off the living room lights", :neutral},
      {"Goodbye, talk to you later", :neutral},
      {"I absolutely love this new feature, it's incredible!", :positive},
      {"This is really annoying and I'm fed up with it.", :negative}
    ]

    sentiments =
      Enum.map(sentiment_samples, fn {text, expected} ->
        log_input.("classify_sentiment", text)

        {result, _} =
          timed.("classify_sentiment(#{String.slice(text, 0, 25)}...)", fn ->
            Brain.ML.SentimentClassifierSimple.classify(text)
          end)

        log_output.("sentiment", result)

        case result do
          {:ok, %{label: label, confidence: conf}} ->
            %{text: text, sentiment: label, confidence: conf, expected: expected}

          _ ->
            %{text: text, sentiment: :neutral, confidence: 0.5, expected: expected}
        end
      end)

    IO.puts("\n  Accumulated sentiments:")
    correct = Enum.count(sentiments, fn s -> s.sentiment == s.expected end)
    total = length(sentiments)

    Enum.each(sentiments, fn s ->
      match = if s.sentiment == s.expected, do: "MATCH", else: "MISMATCH"
      IO.puts("    [#{match}] \"#{String.slice(s.text, 0, 45)}\"")

      IO.puts(
        "      actual=#{s.sentiment} (#{Float.round(s.confidence * 100, 1)}%), expected=#{s.expected}"
      )
    end)

    IO.puts(
      "\n  Sentiment accuracy: #{correct}/#{total} (#{Float.round(correct / max(total, 1) * 100, 1)}%)"
    )

    %{acc | sentiments: sentiments}
  end)

# ============================================================
# 4. Analysis Pipeline
# ============================================================

acc =
  section.("4. Analysis Pipeline", acc, fn acc ->
    text =
      "Good morning! Can you check the weather in Denver for me? Also, I'd love to hear some jazz music. Oh and remind me to call Sarah at noon."

    opts = [world_id: acc.world_id]

    expected_chunk_intents = [
      {"Good morning", "smalltalk.greetings.goodmorning"},
      {"weather in Denver", "weather.query"},
      {"jazz music", "music.play"},
      {"remind me", "reminder.create"}
    ]

    log_input.("Pipeline.process", %{text: text, opts: opts})

    {result, _} =
      timed.("Pipeline.process", fn ->
        Pipeline.process(text, opts)
      end)

    summary = Pipeline.summarize(result)
    log_output.("Pipeline.summarize", summary)

    IO.puts("\n  Pipeline breakdown:")
    IO.puts("    Chunks: #{summary.chunks}")
    IO.puts("    Strategy: #{inspect(summary.overall_strategy)}")

    correct =
      Enum.with_index(summary.analyses)
      |> Enum.reduce(0, fn {a, idx}, correct_count ->
        {_hint, expected} = Enum.at(expected_chunk_intents, idx, {"?", "?"})
        match = if to_string(a.intent) == expected, do: "MATCH", else: "MISMATCH"
        IO.puts("    [#{match}] chunk \"#{a.text}\"")

        IO.puts(
          "      actual=#{inspect(a.intent)}, expected=#{inspect(expected)}, strategy=#{inspect(a.strategy)}"
        )

        if to_string(a.intent) == expected, do: correct_count + 1, else: correct_count
      end)

    total = length(summary.analyses)

    IO.puts(
      "\n  Pipeline intent accuracy: #{correct}/#{total} (#{Float.round(correct / max(total, 1) * 100, 1)}%)"
    )

    %{acc | pipeline_result: result}
  end)

# ============================================================
# 5. Memory Store
# ============================================================

acc =
  section.("5. Memory Store", acc, fn acc ->
    ready = MemoryStore.ready?()
    IO.puts("  Memory.Store ready? #{ready}")

    stats = MemoryStore.stats(world_id: acc.world_id)
    log_output.("stats", stats)

    pipeline_text =
      if acc.pipeline_result do
        acc.pipeline_result.raw_input
      else
        "Good morning! Can you check the weather in Denver for me?"
      end

    intent_summary =
      acc.intents
      |> Enum.map(fn i -> "#{i.intent}(#{Float.round(i.confidence * 100, 1)}%)" end)
      |> Enum.join(", ")

    entity_tags =
      acc.entities
      |> Enum.flat_map(fn e ->
        Enum.map(e.entities, fn ent -> to_string(Map.get(ent, :type, "entity")) end)
      end)
      |> Enum.uniq()

    tags = ["exercise_script" | entity_tags]

    episode_fields = %{
      state: pipeline_text,
      action: "exercise_script",
      outcome: "Intents classified: #{intent_summary}",
      tags: tags
    }

    log_input.("add_episode", episode_fields)

    {_add_result, _} =
      timed.("add_episode", fn ->
        MemoryStore.add_episode(
          episode_fields.state,
          episode_fields.action,
          episode_fields.outcome,
          episode_fields.tags,
          world_id: acc.world_id
        )
      end)

    query_text =
      case Enum.at(acc.intents, 3) do
        %{text: t} ->
          t

        _ ->
          case Enum.at(acc.intents, 1) do
            %{text: t} -> t
            _ -> "remind me about my meeting"
          end
      end

    log_input.("query_similar", %{text: query_text, k: 3, world_id: acc.world_id})

    {similar, _} =
      timed.("query_similar", fn ->
        MemoryStore.query_similar(query_text, 3, world_id: acc.world_id)
      end)

    log_output.("similar episodes", similar)

    top_memory =
      case similar do
        [first | _] -> first
        _ -> nil
      end

    IO.puts("\n  Top retrieved memory: #{inspect(top_memory, limit: 5, pretty: true)}")

    memories = if top_memory, do: [top_memory], else: []
    %{acc | memories: memories}
  end)

# ============================================================
# 6. Enrichment Pipeline
# ============================================================

acc =
  section.("6. Enrichment Pipeline", acc, fn acc ->
    IO.puts("  --- Registered services ---")

    {services, _} =
      timed.("Dispatcher.list_services", fn ->
        Dispatcher.list_services(world: acc.world_id)
      end)

    Enum.each(services, fn svc ->
      IO.puts(
        "    - #{svc.name} (#{svc.display_name}): configured=#{svc.configured}, enabled=#{svc.enabled}"
      )

      IO.puts("      intents: #{inspect(svc.supported_intents)}")
      IO.puts("      provides: #{inspect(svc.provides_fields)}")
    end)

    IO.puts("\n  --- Weather service availability ---")

    {weather_available, _} =
      timed.("Dispatcher.service_available?(:weather)", fn ->
        Dispatcher.service_available?(:weather, world: acc.world_id)
      end)

    IO.puts("  Weather service available? #{weather_available}")

    IO.puts("\n  --- Enricher.prepare_context (weather.query) ---")
    slots = %{location: "Denver"}
    context = %{world_id: acc.world_id}

    log_input.("Enricher.prepare_context", %{
      intent: "weather.query",
      slots: slots,
      context: context
    })

    {enriched_context, _} =
      timed.("Enricher.prepare_context", fn ->
        Enricher.prepare_context("weather.query", slots, context)
      end)

    enrichment_status = Map.get(enriched_context, :enrichment_status)
    enriched_data = Map.get(enriched_context, :enriched_data, %{})
    IO.puts("  Enrichment status: #{inspect(enrichment_status)}")
    log_output.("enriched_data", enriched_data)

    IO.puts("\n  --- Enricher.enrich_response (placeholder substitution) ---")
    template = "The weather is $temperature and $conditions in $location_name"
    log_input.("Enricher.enrich_response", %{template: template})

    {enrich_result, _} =
      timed.("Enricher.enrich_response", fn ->
        Enricher.enrich_response(template, enriched_context)
      end)

    log_output.("enrich_response", enrich_result)

    IO.puts("\n  --- Enrichment metadata ---")
    metadata = Enricher.get_enrichment_metadata(enriched_context)
    log_output.("enrichment_metadata", metadata)

    IO.puts("\n  --- Enricher.prepare_context (music.play -- no enrichment service) ---")
    music_slots = %{genre: "jazz"}

    log_input.("Enricher.prepare_context", %{
      intent: "music.play",
      slots: music_slots,
      context: context
    })

    {music_context, _} =
      timed.("Enricher.prepare_context (music.play)", fn ->
        Enricher.prepare_context("music.play", music_slots, context)
      end)

    IO.puts("  Music enrichment status: #{inspect(Map.get(music_context, :enrichment_status))}")

    IO.puts("\n  --- Dispatcher.dispatch (reminder.create -- no handler) ---")

    log_input.("Dispatcher.dispatch", %{
      intent: "reminder.create",
      slots: %{time: "3pm", task: "meeting"}
    })

    {dispatch_result, _} =
      timed.("Dispatcher.dispatch (reminder.create)", fn ->
        Dispatcher.dispatch("reminder.create", %{time: "3pm", task: "meeting"}, context)
      end)

    log_output.("dispatch result", dispatch_result)

    %{
      acc
      | enrichment: %{
          status: enrichment_status,
          data: enriched_data,
          metadata: metadata,
          context: enriched_context
        }
    }
  end)

# ============================================================
# 7. Response Generator
# ============================================================

acc =
  section.("7. Response Generator", acc, fn acc ->
    gen_inputs =
      acc.intents
      |> Enum.with_index()
      |> Enum.map(fn {intent_entry, idx} ->
        entity_entry = Enum.at(acc.entities, idx) || %{entities: []}

        %{
          intent: to_string(intent_entry.intent),
          entities: entity_entry.entities,
          query_text: intent_entry.text
        }
      end)
      |> Enum.take(4)

    # --- generate/3 across varied intents ---
    IO.puts("  --- Generator.generate (basic, multiple intents) ---")

    gen_results =
      Enum.flat_map(gen_inputs, fn %{intent: intent, entities: entities, query_text: query_text} ->
        log_input.("Generator.generate", %{
          intent: intent,
          entities: entities,
          query_text: query_text
        })

        {result, _} =
          timed.("Generator.generate(#{intent})", fn ->
            Generator.generate(intent, entities, query_text)
          end)

        log_output.("result", result)
        [result]
      end)

    # --- generate_with_events/4 (enrichment-aware path, first input) ---
    IO.puts("\n  --- Generator.generate_with_events (enrichment-aware) ---")
    first = Enum.at(gen_inputs, 0) || %{intent: "unknown", entities: [], query_text: "hello"}

    log_input.("Generator.generate_with_events", %{
      intent: first.intent,
      entities: first.entities,
      query_text: first.query_text,
      events: []
    })

    {events_result, _} =
      timed.("Generator.generate_with_events", fn ->
        Generator.generate_with_events(first.intent, first.entities, first.query_text, [])
      end)

    log_output.("generate_with_events result", events_result)

    gen_results = gen_results ++ [events_result]

    IO.puts("\n  Response source tags:")

    Enum.each(gen_results, fn
      {:ok, _response, source_tag} ->
        IO.puts("    - #{inspect(source_tag)}")

      other ->
        IO.puts("    - (non-standard result: #{inspect(other, limit: 3)})")
    end)

    %{acc | gen_responses: gen_results}
  end)

# ============================================================
# 8. Brain Evaluate -- Multiple Conversations
# ============================================================

acc =
  section.("8. Brain Evaluate (Multi-Conversation)", acc, fn acc ->
    run_conversation = fn label, turns, acc_inner ->
      IO.puts("\n  --- Conversation #{label} ---")
      IO.puts("  Creating conversation with world_id: #{acc_inner.world_id}")

      {:ok, conv_id} = Brain.create_conversation(world_id: acc_inner.world_id)
      IO.puts("  Conversation ID: #{conv_id}")

      turn_results =
        Enum.with_index(turns, 1)
        |> Enum.map(fn {{turn_input, expected_intent}, turn_num} ->
          IO.puts("\n    -- #{label} Turn #{turn_num} --")

          log_input.("Brain.evaluate [#{label}/#{turn_num}]", %{
            conversation_id: conv_id,
            input: turn_input,
            expected_intent: expected_intent
          })

          {result, elapsed} =
            timed.("Brain.evaluate [#{label}/#{turn_num}]", fn ->
              Brain.evaluate(conv_id, turn_input)
            end)

          eval_data = log_eval_result.("#{label}/#{turn_num}", result)

          actual_intent =
            if is_map(eval_data) and Map.has_key?(eval_data, :context) do
              get_in(eval_data, [:context, :intent])
            end

          match =
            cond do
              expected_intent == nil -> :skip
              to_string(actual_intent) == expected_intent -> :match
              true -> :mismatch
            end

          case match do
            :match ->
              IO.puts("  [MATCH] intent=#{inspect(actual_intent)}")

            :mismatch ->
              IO.puts(
                "  [MISMATCH] actual=#{inspect(actual_intent)}, expected=#{inspect(expected_intent)}"
              )

            :skip ->
              nil
          end

          %{
            turn: turn_num,
            input: turn_input,
            expected_intent: expected_intent,
            actual_intent: actual_intent,
            intent_match: match,
            result: eval_data,
            elapsed_ms: elapsed,
            processing_method:
              if(is_map(eval_data), do: Map.get(eval_data, :processing_method), else: nil)
          }
        end)

      IO.puts("\n    -- #{label} Conversation Summary --")
      conv = Brain.get_conversation(conv_id)
      msg_count = if is_map(conv), do: length(Map.get(conv, :memory, [])), else: "?"
      IO.puts("    Messages: #{msg_count}")

      methods = turn_results |> Enum.map(& &1.processing_method) |> Enum.filter(& &1)
      IO.puts("    Processing methods: #{inspect(methods)}")

      scorable = Enum.filter(turn_results, &(&1.intent_match in [:match, :mismatch]))
      correct = Enum.count(scorable, &(&1.intent_match == :match))
      IO.puts("    Intent accuracy: #{correct}/#{length(scorable)}")

      Brain.end_conversation(conv_id)
      IO.puts("    Conversation ended.")

      %{label: label, conversation_id: conv_id, turns: turn_results}
    end

    # --- Conversation A: Weather + follow-ups (enrichment + slot carry-over) ---
    conv_a =
      run_conversation.(
        "A (weather + follow-up)",
        [
          {"What's the weather like in Denver?", "weather.query"},
          {"What about tomorrow?", "weather.query"},
          {"And in Tokyo?", "weather.query"}
        ],
        acc
      )

    # --- Conversation B: Music + device control (directives) ---
    conv_b =
      run_conversation.(
        "B (music + device)",
        [
          {"Play some jazz music", "music.play"},
          {"Turn it up a bit", "smarthome.device.volume.up"},
          {"Actually, skip to the next song", "music.player.skip_forward"},
          {"Turn off the lights when the album is done", "smarthome.lights.switch.schedule.off"}
        ],
        acc
      )

    # --- Conversation C: Multi-sentence + mixed domains ---
    pipeline_followup =
      if acc.pipeline_result do
        summary = Pipeline.summarize(acc.pipeline_result)

        intents =
          summary.analyses
          |> Enum.map(fn a -> "#{a.text}: #{inspect(a.intent)}" end)
          |> Enum.join("; ")

        "You mentioned: #{intents}. Can you elaborate on each?"
      else
        "Can you tell me more about that?"
      end

    conv_c =
      run_conversation.(
        "C (multi-sentence)",
        [
          {"Good morning! What's the news today? Also, set a reminder for my dentist appointment at 2pm.",
           "news.query"},
          {pipeline_followup, nil},
          {"Thanks for that. One more thing -- what's the capital of Japan?", "knowledge.capital"}
        ],
        acc
      )

    # --- Conversation D: Sentiment arc (positive -> negative -> recovery) ---
    conv_d =
      run_conversation.(
        "D (sentiment arc)",
        [
          {"I'm really excited to learn about machine learning!", "smalltalk.user.excited"},
          {"This is so frustrating, nothing is working right.", "smalltalk.user.angry"},
          {"Actually, I think I figured it out. Can you help me understand neural networks?",
           "knowledge.define"},
          {"You've been really helpful, thank you!", "smalltalk.appraisal.thank_you"}
        ],
        acc
      )

    # --- Conversation E: Knowledge + meta + memory ---
    memory_input =
      case Enum.at(acc.memories, 0) do
        nil ->
          "What have you learned recently?"

        mem ->
          state = Map.get(mem, :state, Map.get(mem, "state", ""))

          if is_binary(state) and String.length(state) > 0 do
            String.slice(state, 0, 200)
          else
            "What have you learned recently?"
          end
      end

    conv_e =
      run_conversation.(
        "E (knowledge + meta)",
        [
          {"What do you know about yourself?", "meta.self_knowledge"},
          {"What is the definition of polymorphism?", "knowledge.define"},
          {"What topics have we discussed?", "meta.memory_check"},
          {memory_input, nil}
        ],
        acc
      )

    # --- Conversation F: Practical tasks (calendar, reminders, navigation) ---
    conv_f =
      run_conversation.(
        "F (practical tasks)",
        [
          {"Schedule a meeting with the team for Friday at 10am", "calendar.schedule"},
          {"Set a reminder to buy groceries this evening", "reminder.create"},
          {"How do I get to the nearest coffee shop?", "navigation.directions"},
          {"Cancel that meeting actually, something came up", "calendar.cancel"}
        ],
        acc
      )

    all_turns =
      [conv_a, conv_b, conv_c, conv_d, conv_e, conv_f]
      |> Enum.flat_map(fn conv ->
        Enum.map(conv.turns, &Map.put(&1, :conversation, conv.label))
      end)

    %{acc | eval_turns: all_turns}
  end)

# ============================================================
# 9. KG Signal Strengthening — PredicateNormalizer
# ============================================================

acc =
  section.("9. KG Signals — PredicateNormalizer", acc, fn acc ->
    alias Brain.ML.KnowledgeGraph.PredicateNormalizer

    IO.puts("  Canonical relations: #{length(PredicateNormalizer.canonical_relations())}")

    IO.puts(
      "  Sample canonicals: #{inspect(Enum.take(PredicateNormalizer.canonical_relations(), 10))}"
    )

    test_predicates = [
      {"is_a", :mapped, "IsA"},
      {"IsA", :exact, "IsA"},
      {"LOCATED_AT", :mapped, "AtLocation"},
      {"located_in", :mapped, "AtLocation"},
      {"likes", :mapped, "Likes"},
      {"wants", :mapped, "Wants"},
      {"made_by", :mapped, "MadeBy"},
      {"CreatedBy", :mapped, "MadeBy"},
      {"xyzzy_nonsense", :oov, nil},
      {"", :empty, nil}
    ]

    results =
      Enum.map(test_predicates, fn {pred, expected_kind, expected_canon} ->
        result = PredicateNormalizer.normalize(pred)

        {ok, actual_kind, actual_canon} =
          case result do
            {:ok, canon, kind} -> {true, kind, canon}
            {:error, :oov} -> {true, :oov, nil}
            {:error, :empty} -> {true, :empty, nil}
            other -> {false, :unexpected, inspect(other)}
          end

        pass =
          case expected_kind do
            :oov -> actual_kind == :oov
            :empty -> actual_kind == :empty
            _ -> actual_canon == expected_canon
          end

        status = if pass, do: "PASS", else: "FAIL"
        IO.puts("    #{status}: normalize(#{inspect(pred)}) => #{inspect(result)}")

        %{predicate: pred, result: result, pass: pass}
      end)

    pass_count = Enum.count(results, & &1.pass)
    total = length(results)
    IO.puts("\n  PredicateNormalizer: #{pass_count}/#{total} passed")

    if pass_count < total do
      IO.puts("  [WARN] Some normalizer tests failed — check alias JSON")
    end

    Map.put(acc, :kg_normalizer_results, results)
  end)

# ============================================================
# 10. KG Signal Strengthening — TripleScorer Status
# ============================================================

acc =
  section.("10. KG Signals — TripleScorer Status", acc, fn acc ->
    alias Brain.ML.KnowledgeGraph.TripleScorer

    scorer_ready = TripleScorer.ready?()
    IO.puts("  TripleScorer ready? #{scorer_ready}")

    model_version =
      case TripleScorer.current_model_version() do
        {:ok, v} ->
          IO.puts("  Model version: #{v}")
          v

        {:error, reason} ->
          IO.puts("  Model version: unavailable (#{inspect(reason)})")
          nil
      end

    relation_coverage =
      case TripleScorer.relation_coverage() do
        {:ok, coverage} when is_map(coverage) ->
          IO.puts("  Relation coverage: #{map_size(coverage)} relations")
          top_5 = coverage |> Enum.sort_by(fn {_, c} -> -c end) |> Enum.take(5)

          Enum.each(top_5, fn {rel, count} ->
            IO.puts("    - #{rel}: #{count} training examples")
          end)

          coverage

        {:error, reason} ->
          IO.puts("  Relation coverage: unavailable (#{inspect(reason)})")
          %{}
      end

    if scorer_ready do
      IO.puts("\n  --- Scoring sample triples ---")

      sample_triples = [
        {"dog", "IsA", "animal"},
        {"paris", "AtLocation", "france"},
        {"tesla", "MadeBy", "elon_musk"},
        {"banana", "IsA", "vehicle"},
        {"alice", "Visited", "berlin"}
      ]

      Enum.each(sample_triples, fn {h, r, t} ->
        {result, elapsed} =
          timed.("score(#{h}, #{r}, #{t})", fn ->
            TripleScorer.score(h, r, t)
          end)

        case result do
          {:ok, score} ->
            quality =
              cond do
                score >= 0.7 -> "HIGH"
                score >= 0.4 -> "MEDIUM"
                true -> "LOW"
              end

            IO.puts(
              "    (#{h}, #{r}, #{t}) => #{Float.round(score, 4)} [#{quality}] (#{elapsed}ms)"
            )

          {:error, reason} ->
            IO.puts("    (#{h}, #{r}, #{t}) => ERROR: #{inspect(reason)} (#{elapsed}ms)")
        end
      end)

      IO.puts("\n  --- Batch scoring ---")
      batch = Enum.map(sample_triples, fn {h, r, t} -> {h, r, t} end)

      {batch_result, elapsed} =
        timed.("score_batch(#{length(batch)} triples)", fn ->
          TripleScorer.score_batch(batch)
        end)

      case batch_result do
        {:ok, scores} ->
          IO.puts("    Batch scored #{length(scores)} triples in #{elapsed}ms")

          Enum.zip(sample_triples, scores)
          |> Enum.each(fn {{h, r, t}, score} ->
            IO.puts("      (#{h}, #{r}, #{t}) => #{Float.round(score, 4)}")
          end)

        {:error, reason} ->
          IO.puts("    Batch scoring failed: #{inspect(reason)}")
      end
    else
      IO.puts("\n  [SKIP] TripleScorer not ready — scoring tests skipped")
      IO.puts("  Run `mix train_kg_lstm` to train the model")
    end

    acc
    |> Map.put(:kg_scorer_ready, scorer_ready)
    |> Map.put(:kg_model_version, model_version)
    |> Map.put(:kg_relation_count, map_size(relation_coverage))
  end)

# ============================================================
# 11. KG Signal Strengthening — EntityVectorCache
# ============================================================

acc =
  section.("11. KG Signals — EntityVectorCache", acc, fn acc ->
    alias Brain.ML.KnowledgeGraph.EntityVectorCache

    cache_stats = EntityVectorCache.stats()
    IO.puts("  Cache stats: #{inspect(cache_stats)}")

    if acc[:kg_scorer_ready] do
      IO.puts("\n  --- Computing entity embeddings ---")

      test_entities = ["dog", "cat", "paris", "python", "coffee"]

      vectors =
        Enum.map(test_entities, fn entity ->
          {result, elapsed} =
            timed.("get_or_compute(#{entity})", fn ->
              EntityVectorCache.get_or_compute(acc.world_id, entity)
            end)

          case result do
            {:ok, tensor} ->
              dims = Nx.shape(tensor) |> Tuple.to_list() |> List.last()
              IO.puts("    #{entity}: #{dims}-dim vector (#{elapsed}ms)")
              {entity, tensor}

            {:error, reason} ->
              IO.puts("    #{entity}: ERROR #{inspect(reason)} (#{elapsed}ms)")
              nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      if length(vectors) >= 2 do
        IO.puts("\n  --- Pairwise cosine similarity ---")
        [{e1, v1}, {e2, v2} | _] = vectors
        sim = FourthWall.Math.cosine_similarity(Nx.to_flat_list(v1), Nx.to_flat_list(v2))
        IO.puts("    cos(#{e1}, #{e2}) = #{Float.round(sim, 4)}")

        if length(vectors) >= 3 do
          {e3, v3} = Enum.at(vectors, 2)
          sim13 = FourthWall.Math.cosine_similarity(Nx.to_flat_list(v1), Nx.to_flat_list(v3))
          sim23 = FourthWall.Math.cosine_similarity(Nx.to_flat_list(v2), Nx.to_flat_list(v3))
          IO.puts("    cos(#{e1}, #{e3}) = #{Float.round(sim13, 4)}")
          IO.puts("    cos(#{e2}, #{e3}) = #{Float.round(sim23, 4)}")
        end
      end

      IO.puts("\n  --- Cache hit test (second lookup) ---")
      first_entity = hd(test_entities)

      {_result2, elapsed2} =
        timed.("get_or_compute(#{first_entity}) [cached]", fn ->
          EntityVectorCache.get_or_compute(acc.world_id, first_entity)
        end)

      IO.puts("    Second lookup: #{elapsed2}ms (should be near-zero if cached)")

      stats_after = EntityVectorCache.stats()

      IO.puts(
        "\n  Cache after: size=#{stats_after.size}, hits=#{stats_after.hits}, misses=#{stats_after.misses}"
      )
    else
      IO.puts("\n  [SKIP] TripleScorer not ready — entity vectors unavailable")
    end

    acc
  end)

# ============================================================
# 12. KG Signal Strengthening — ConsolidationBridge
# ============================================================

acc =
  section.("12. KG Signals — ConsolidationBridge", acc, fn acc ->
    alias Brain.Epistemic.ConsolidationBridge

    IO.puts("  --- Testing triple extraction from text ---")

    test_sentences = [
      "Alice visited Berlin last summer",
      "Dogs are a type of animal",
      "Tesla was made by Elon Musk",
      "The conference is at the convention center"
    ]

    Enum.each(test_sentences, fn text ->
      fake_sf = %Brain.Memory.Types.SemanticFact{
        id: "test_#{:erlang.phash2(text)}",
        timestamp: DateTime.utc_now(),
        representation: text,
        evidence_ids: [],
        embedding: [],
        tags: []
      }

      triples = ConsolidationBridge.extract_triples(fake_sf)
      IO.puts("    \"#{text}\"")

      if triples == [] do
        IO.puts("      => (no triples extracted)")
      else
        Enum.each(triples, fn {s, p, o} ->
          norm = Brain.ML.KnowledgeGraph.PredicateNormalizer.normalize(p)

          norm_str =
            case norm do
              {:ok, canon, kind} -> "#{canon} (#{kind})"
              {:error, reason} -> "OOV (#{reason})"
            end

          IO.puts("      => (#{s}, #{p}, #{o}) normalized: #{norm_str}")
        end)
      end
    end)

    IO.puts("\n  --- KG signals config ---")
    kg_config = Application.get_env(:brain, :kg_signals, [])
    IO.puts("    enabled:                #{Keyword.get(kg_config, :enabled, true)}")
    IO.puts("    srl_gating:             #{Keyword.get(kg_config, :srl_gating, true)}")
    IO.puts("    consolidation_blend:    #{Keyword.get(kg_config, :consolidation_blend, 0.6)}")
    IO.puts("    memory_rerank:          #{Keyword.get(kg_config, :memory_rerank, true)}")
    IO.puts("    novelty_downweight:     #{Keyword.get(kg_config, :novelty_downweight, true)}")

    IO.puts(
      "    contradiction_default:  #{Keyword.get(kg_config, :contradiction_default_kg, true)}"
    )

    IO.puts(
      "    entity_promoter_gate:   #{Keyword.get(kg_config, :entity_promoter_kg_gate, true)}"
    )

    acc
  end)

# ============================================================
# 13. KG Signal Strengthening — NoveltyDetector KG Downweight
# ============================================================

acc =
  section.("13. KG Signals — NoveltyDetector KG Downweight", acc, fn acc ->
    alias Brain.Analysis.NoveltyDetector

    IO.puts("  --- Novelty detection with KG awareness ---")

    test_cases = [
      %{
        text: "Quantum computing uses qubits",
        best_score: 0.3,
        margin: 0.1,
        label: "low confidence + small margin => novel"
      },
      %{
        text: "Hello there",
        best_score: 0.9,
        margin: 0.5,
        label: "high confidence + large margin => not novel"
      },
      %{
        text: "The Eiffel Tower is located in Paris",
        best_score: 0.35,
        margin: 0.15,
        label: "factual, low confidence => novel (maybe KG-downweighted)"
      }
    ]

    Enum.each(test_cases, fn tc ->
      result =
        NoveltyDetector.is_novel?(tc.best_score, tc.margin,
          text: tc.text,
          entities: []
        )

      case result do
        {:novel, score} ->
          IO.puts("    #{tc.label}")
          IO.puts("      text: \"#{tc.text}\"")
          IO.puts("      novelty_score: #{Float.round(score, 4)}")

        :not_novel ->
          IO.puts("    #{tc.label}")
          IO.puts("      text: \"#{tc.text}\"")
          IO.puts("      result: not novel")
      end
    end)

    IO.puts("\n  --- KG downweight function directly ---")
    raw_score = 0.85
    downweighted = NoveltyDetector.maybe_kg_downweight(raw_score, "Dogs are animals", nil)
    IO.puts("    raw=#{raw_score}, after KG downweight=#{Float.round(downweighted, 4)}")

    if downweighted < raw_score do
      IO.puts(
        "    [OK] Downweight applied (reduction: #{Float.round((raw_score - downweighted) * 100, 1)}%)"
      )
    else
      IO.puts("    [INFO] No downweight applied (no matching beliefs or KG signals disabled)")
    end

    acc
  end)

# ============================================================
# 14. KG Signal Strengthening — Memory Re-rank Verification
# ============================================================

acc =
  section.("14. KG Signals — Memory Re-rank", acc, fn acc ->
    IO.puts("  --- Memory retrieval with KG re-rank ---")

    IO.puts(
      "  memory_rerank enabled? #{inspect(Application.get_env(:brain, :kg_signals, []) |> Keyword.get(:memory_rerank, true))}"
    )

    queries = [
      "What is the weather forecast?",
      "Tell me about machine learning",
      "How do I get to the airport?"
    ]

    Enum.each(queries, fn query ->
      {result, elapsed} =
        timed.("query_similar(#{String.slice(query, 0, 30)}...)", fn ->
          MemoryStore.query_similar(query, 3, world_id: acc.world_id)
        end)

      case result do
        {:ok, episodes} when is_list(episodes) ->
          IO.puts("    \"#{query}\" => #{length(episodes)} results (#{elapsed}ms)")

          Enum.each(Enum.take(episodes, 2), fn {ep, sim} ->
            state = Map.get(ep, :state, "") |> String.slice(0, 60)
            IO.puts("      - sim=#{Float.round(sim, 4)}: \"#{state}...\"")
          end)

        {:ok, []} ->
          IO.puts("    \"#{query}\" => 0 results (#{elapsed}ms)")

        {:error, reason} ->
          IO.puts("    \"#{query}\" => ERROR: #{inspect(reason)} (#{elapsed}ms)")

        other ->
          IO.puts("    \"#{query}\" => unexpected: #{inspect(other, limit: 3)}")
      end
    end)

    acc
  end)

# ============================================================
# 15. KG Signal Strengthening — StanceTracker Canonicalization
# ============================================================

acc =
  section.("15. KG Signals — StanceTracker Topic Canonicalization", acc, fn acc ->
    alias Brain.Epistemic.StanceTracker

    ready = StanceTracker.ready?()
    IO.puts("  StanceTracker ready? #{ready}")

    if ready do
      IO.puts("\n  --- Recording stances with variant topic phrasings ---")

      {:ok, conv_id} = Brain.create_conversation(world_id: acc.world_id)

      topic_variants = [
        {"weather", 0.5},
        {"weather forecast", 0.6},
        {"climate", -0.2}
      ]

      Enum.each(topic_variants, fn {topic, position} ->
        StanceTracker.record_stance(conv_id, topic, position, :user)
        IO.puts("    Recorded: topic=#{inspect(topic)}, position=#{position}")
      end)

      {:ok, stances} = StanceTracker.conversation_stances(conv_id)
      IO.puts("\n  Stored topics (should be canonicalized):")

      Enum.each(stances, fn {topic, observations} ->
        IO.puts("    #{inspect(topic)}: #{length(observations)} observation(s)")
      end)

      canonical_count = map_size(stances)
      IO.puts("\n  Unique topics stored: #{canonical_count}")

      if canonical_count < length(topic_variants) do
        IO.puts("  [OK] Topic canonicalization merged some variants")
      else
        IO.puts("  [INFO] No merging occurred (vectors may not be available)")
      end

      Brain.end_conversation(conv_id)
    else
      IO.puts("  [SKIP] StanceTracker not ready")
    end

    acc
  end)

# ============================================================
# 16. KG Signal Strengthening — Predicate Vocab Audit Summary
# ============================================================

acc =
  section.("16. KG Signals — Predicate Vocabulary Coverage", acc, fn acc ->
    alias Brain.ML.KnowledgeGraph.PredicateNormalizer

    IO.puts("  --- Coverage of known predicate sources ---")

    srl_predicates = ~w(LOCATED_AT OCCURRED_AT CAUSED_BY MANNER PURPOSE)

    epistemic_predicates =
      ~w(likes wants needs interested_in consolidated_knowledge claims believes)

    hierarchy_predicates = ~w(is_a has_subtype has_type made_by located_in)

    sources = [
      {"SRL roles", srl_predicates},
      {"Epistemic atoms", epistemic_predicates},
      {"Hierarchy relations", hierarchy_predicates}
    ]

    total_mapped = 0
    total_oov = 0

    {total_mapped, total_oov} =
      Enum.reduce(sources, {0, 0}, fn {source_name, predicates}, {mapped, oov} ->
        results = Enum.map(predicates, fn p -> {p, PredicateNormalizer.normalize(p)} end)

        mapped_count = Enum.count(results, fn {_, r} -> match?({:ok, _, _}, r) end)
        oov_count = Enum.count(results, fn {_, r} -> r == {:error, :oov} end)

        IO.puts(
          "    #{source_name}: #{mapped_count}/#{length(predicates)} mapped, #{oov_count} OOV"
        )

        oov_preds =
          results
          |> Enum.filter(fn {_, r} -> r == {:error, :oov} end)
          |> Enum.map(fn {p, _} -> p end)

        if oov_preds != [] do
          IO.puts("      OOV: #{inspect(oov_preds)}")
        end

        {mapped + mapped_count, oov + oov_count}
      end)

    total = total_mapped + total_oov
    coverage_pct = if total > 0, do: Float.round(total_mapped / total * 100, 1), else: 0.0
    IO.puts("\n  Overall coverage: #{total_mapped}/#{total} (#{coverage_pct}%)")

    acc
  end)

# ============================================================
# Final Summary
# ============================================================

divider.("FINAL SUMMARY")

IO.puts("  World ID:              #{acc.world_id}")
IO.puts("  Intents classified:    #{length(acc.intents)}")
IO.puts("  Entity extractions:    #{length(acc.entities)}")
IO.puts("  Sentiment analyses:    #{length(acc.sentiments)}")
IO.puts("  Memories stored:       #{length(acc.memories)}")

enrichment_status = if acc.enrichment, do: inspect(acc.enrichment.status), else: "not run"
IO.puts("  Enrichment status:     #{enrichment_status}")

IO.puts("  Generator responses:   #{length(acc.gen_responses)}")

# --- Classification accuracy ---
intent_correct = Enum.count(acc.intents, fn i -> to_string(i.intent) == i[:expected] end)
intent_total = length(acc.intents)
IO.puts("\n  --- Classification Accuracy ---")

IO.puts(
  "  Standalone intent:     #{intent_correct}/#{intent_total} (#{Float.round(intent_correct / max(intent_total, 1) * 100, 1)}%)"
)

sentiment_correct = Enum.count(acc.sentiments, fn s -> s.sentiment == s[:expected] end)
sentiment_total = length(acc.sentiments)

IO.puts(
  "  Sentiment:             #{sentiment_correct}/#{sentiment_total} (#{Float.round(sentiment_correct / max(sentiment_total, 1) * 100, 1)}%)"
)

conversations =
  acc.eval_turns
  |> Enum.group_by(& &1.conversation)

conv_scorable = Enum.filter(acc.eval_turns, &(&1[:intent_match] in [:match, :mismatch]))
conv_correct = Enum.count(conv_scorable, &(&1[:intent_match] == :match))

IO.puts(
  "  Conversation intent:   #{conv_correct}/#{length(conv_scorable)} (#{Float.round(conv_correct / max(length(conv_scorable), 1) * 100, 1)}%)"
)

IO.puts("\n  Conversations:         #{map_size(conversations)}")
IO.puts("  Total eval turns:      #{length(acc.eval_turns)}")

IO.puts("\n  Per-conversation breakdown:")

Enum.each(conversations, fn {label, turns} ->
  methods = turns |> Enum.map(& &1.processing_method) |> Enum.filter(& &1) |> Enum.map(&inspect/1)
  total_ms = Enum.reduce(turns, 0, fn t, sum -> sum + (t.elapsed_ms || 0) end)
  scorable = Enum.filter(turns, &(&1[:intent_match] in [:match, :mismatch]))
  correct = Enum.count(scorable, &(&1[:intent_match] == :match))

  IO.puts(
    "    #{label}: #{length(turns)} turns, #{total_ms}ms total, intent #{correct}/#{length(scorable)}"
  )

  IO.puts("      methods: #{Enum.join(methods, ", ")}")

  mismatches = Enum.filter(turns, &(&1[:intent_match] == :mismatch))

  Enum.each(mismatches, fn t ->
    IO.puts(
      "      MISMATCH turn #{t.turn}: actual=#{inspect(t[:actual_intent])}, expected=#{inspect(t[:expected_intent])}"
    )
  end)
end)

# --- KG Signal Strengthening summary ---
IO.puts("\n  --- KG Signal Strengthening ---")
IO.puts("  TripleScorer ready:    #{acc[:kg_scorer_ready] || false}")
IO.puts("  Model version:         #{acc[:kg_model_version] || "not loaded"}")
IO.puts("  Trained relations:     #{acc[:kg_relation_count] || 0}")

normalizer_results = acc[:kg_normalizer_results] || []

if normalizer_results != [] do
  pass = Enum.count(normalizer_results, & &1.pass)
  IO.puts("  Normalizer tests:      #{pass}/#{length(normalizer_results)} passed")
end

kg_config = Application.get_env(:brain, :kg_signals, [])
IO.puts("  KG signals enabled:    #{Keyword.get(kg_config, :enabled, true)}")

IO.puts(
  "  Active features:       #{Enum.count(kg_config, fn {_k, v} -> v == true end)}/#{length(kg_config)}"
)

cache_stats =
  try do
    Brain.ML.KnowledgeGraph.EntityVectorCache.stats()
  rescue
    _ -> %{size: 0, hits: 0, misses: 0}
  end

IO.puts("  Entity cache size:     #{Map.get(cache_stats, :size, 0)}")

IO.puts(
  "  Cache hit/miss:        #{Map.get(cache_stats, :hits, 0)}/#{Map.get(cache_stats, :misses, 0)}"
)

if acc.failed == [] do
  IO.puts("\n  Failed sections:       none")
else
  IO.puts("\n  Failed sections:       #{Enum.join(Enum.reverse(acc.failed), ", ")}")
end

IO.puts("\n" <> String.duplicate("=", 72))
IO.puts("  Exercise complete.")
IO.puts(String.duplicate("=", 72) <> "\n")
