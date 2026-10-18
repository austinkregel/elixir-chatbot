defmodule Brain.Response.DiscoursePlannerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.DiscoursePlanner
  alias Brain.Analysis.{InternalModel, ChunkAnalysis, SpeechActResult}

  defp build_model(text, opts \\ []) do
    speech_act = Keyword.get(opts, :speech_act, SpeechActResult.new(:directive, :question, 0.8, is_question: true))
    sentiment = Keyword.get(opts, :sentiment, %{label: :neutral, confidence: 0.5})
    confidence = Keyword.get(opts, :confidence, 0.8)
    response_strategy = Keyword.get(opts, :response_strategy, :can_respond)

    analysis = %ChunkAnalysis{
      chunk_index: 0,
      text: text,
      speech_act: speech_act,
      intent: Keyword.get(opts, :intent, "general.query"),
      entities: Keyword.get(opts, :entities, []),
      sentiment: sentiment,
      confidence: confidence,
      response_strategy: response_strategy,
      epistemic_status: Keyword.get(opts, :epistemic_status, :unchecked),
      related_beliefs: Keyword.get(opts, :related_beliefs, []),
      slots: Keyword.get(opts, :slots, nil)
    }

    %InternalModel{
      raw_input: text,
      analyses: [analysis],
      overall_strategy: response_strategy
    }
  end

  describe "plan/1" do
    test "plans a greeting response" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      primitives = DiscoursePlanner.plan(model)

      assert is_list(primitives)
      assert length(primitives) >= 1

      types = Enum.map(primitives, & &1.type)
      assert :acknowledgment in types
    end

    test "plans a factual question response" do
      model = build_model("What's the weather?",
        speech_act: SpeechActResult.new(:directive, :question, 0.8, is_question: true),
        intent: "weather.query"
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :framing in types or :content in types
    end

    test "plans with attunement for negative sentiment" do
      model = build_model("I had a terrible day",
        speech_act: SpeechActResult.new(:assertive, :statement, 0.7),
        sentiment: %{label: :negative, confidence: 0.8}
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :attunement in types
    end

    test "inserts hedging for low confidence" do
      model = build_model("What's that?",
        speech_act: SpeechActResult.new(:directive, :question, 0.3, is_question: true),
        confidence: 0.2
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :hedging in types
    end

    test "handles contradiction" do
      model = build_model("Actually my name is Bob",
        speech_act: SpeechActResult.new(:assertive, :statement, 0.8),
        epistemic_status: :contradicted
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :contradiction_response in types
    end

    test "seeds content from analysis" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      primitives = DiscoursePlanner.plan(model)
      social_ack = Enum.find(primitives, &(&1.type == :acknowledgment and &1.variant == :social))

      if social_ack do
        assert Map.has_key?(social_ack.content, :speech_act_sub_type)
      end
    end

    test "returns fallback for nil input" do
      primitives = DiscoursePlanner.plan(nil)
      assert is_list(primitives)
      assert length(primitives) >= 1
    end
  end

  describe "plan/2 for what the system produces" do
    alias Brain.Analysis.ChunkProfile
    alias Brain.FactDatabase.Fact

    defp plan_of(model), do: model |> DiscoursePlanner.plan() |> Enum.map(&{&1.type, &1.variant})

    defp with_chunk(model, fields), do: %{model | analyses: Enum.map(model.analyses, &Map.merge(&1, fields))}

    @photosynthesis %Fact{id: "sci_014", entity: "photosynthesis", fact: "Photosynthesis is a process", category: "science"}

    # The intent registry's meta.self_knowledge, a directive requesting
    # information, whose consolidated domain is meta (intent_domains.json).
    test "a self-knowledge query of the meta domain is told what is known, whatever its sentiment" do
      for sentiment <- [:neutral, :negative] do
        model =
          "What do you know about me?"
          |> build_model(
            speech_act: SpeechActResult.new(:directive, :request_information, 0.9, is_question: true),
            intent: "meta.self_knowledge",
            sentiment: %{label: sentiment, confidence: 0.5}
          )
          |> with_chunk(%{profile: %ChunkProfile{domain: :meta}})

        assert plan_of(model) == [{:content, :narrative}, {:follow_up, :correction_invite}]
      end
    end

    test "another meta intent is not planned as a self-knowledge query" do
      model =
        "how are your systems"
        |> build_model(
          speech_act: SpeechActResult.new(:directive, :request_information, 0.9, is_question: true),
          intent: "status.report"
        )
        |> with_chunk(%{profile: %ChunkProfile{domain: :meta}})

      refute {:content, :narrative} in plan_of(model)
    end

    # An explanation is asked for with an imperative ("Explain ..."), a
    # directive requesting information that is no question.
    test "a request for information that is no question, with something known, is explained" do
      model =
        "Explain how photosynthesis works"
        |> build_model(speech_act: SpeechActResult.new(:directive, :request_information, 0.9, is_question: false))
        |> with_chunk(%{related_facts: [@photosynthesis]})

      assert plan_of(model) == [{:framing, :informative}, {:content, :explanatory}]
    end

    test "a question with something known is answered with it, and a polar one by its alignment" do
      question = SpeechActResult.new(:directive, :request_information, 0.9, is_question: true)
      known = %{related_facts: [@photosynthesis]}

      assert "What is photosynthesis?" |> build_model(speech_act: question) |> with_chunk(known) |> plan_of() ==
               [{:framing, :informative}, {:content, :factual}]

      for {alignment, framing} <- [confirming: :affirmative, disconfirming: :negative] do
        aligned = Map.put(known, :fact_alignment, %{alignment: alignment, statement: @photosynthesis.fact})

        assert "Is photosynthesis a process?" |> build_model(speech_act: question) |> with_chunk(aligned) |> plan_of() ==
                 [{:framing, framing}, {:content, :factual}]
      end
    end
  end

  describe "plan/2 when the turn's service returned an error" do
    alias Brain.Response.{ContentSpecifier, Primitive, ResponseEvaluator, SurfaceRealizer}

    # Realizing a service's reply reads no world data; the world is only the
    # one realization requires to be named.
    @world_id "test_service_reply"

    defp failed(reason, service, intent) do
      [
        unified_context: %{
          enrichment: %{
            enrichment_status: :failed,
            enrichment_error: reason,
            enrichment_service: service,
            intent: intent,
            slots: %{},
            enriched_data: %{}
          }
        }
      ]
    end

    defp model_for(text, intent) do
      build_model(text,
        speech_act: SpeechActResult.new(:directive, :command, 0.8, is_imperative: true),
        intent: intent
      )
    end

    # The plan, specified and realized as the refinement loop does.
    defp respond(reason, service, intent, text \\ "do it") do
      model = model_for(text, intent)
      opts = failed(reason, service, intent)
      [analysis] = model.analyses
      plan = DiscoursePlanner.plan(model, opts)
      specified = ContentSpecifier.specify(plan, analysis, opts)

      {:ok, rendered, response, nil} =
        SurfaceRealizer.realize(specified, [world_id: @world_id, analysis: analysis] ++ opts)

      assert Enum.all?(rendered, &(&1.source == :service_reply and &1.rendered == response))
      {plan, specified, response}
    end

    test "several matching events: the plan asks which one, and the reply lists each candidate with its day" do
      events = [%{event: "Dentist", date: "Friday", time: "3 pm"}, %{event: "Standup", date: "", time: "5 pm"}]
      reason = {:ambiguous, :calendar, events}

      {plan, _specified, text} = respond(reason, :local_scheduler, "calendar.cancel")

      assert [%Primitive{type: :follow_up, variant: :clarification, content: content}] = plan
      assert content.reply == :ambiguous_item
      assert content.ambiguity_type == :ambiguous_item
      assert content.item_kind == :calendar
      assert content.candidates == events
      assert content.enrichment_error == reason
      assert content.service_name == "Local Scheduler"

      assert text =~ "Dentist on Friday at 3 pm or Standup at 5 pm"
      assert String.ends_with?(text, "?")
    end

    test "three matching events are listed with the last after the conjunction" do
      events = [%{event: "A", time: "1 pm", date: nil}, %{event: "B", time: nil, date: nil}, %{event: "C", time: "3 pm", date: nil}]

      {_plan, _specified, text} = respond({:ambiguous, :calendar, events}, :local_scheduler, "calendar.cancel")

      assert text =~ "A at 1 pm, B or C at 3 pm"
    end

    test "no event identified: the plan asks which event, listing every event the calendar has" do
      events = [%{event: "Dentist", date: "Friday", time: "3 pm"}, %{event: "Standup", date: "", time: "5 pm"}]
      reason = {:event_not_identified, events}

      {plan, _specified, text} = respond(reason, :local_scheduler, "calendar.cancel")

      assert [%Primitive{type: :follow_up, variant: :clarification, content: content}] = plan
      assert content.reply == :ambiguous_item
      assert content.item_kind == :calendar
      assert content.candidates == events
      assert content.enrichment_error == reason
      refute Map.has_key?(content, :missing_slots)

      assert text =~ "Dentist on Friday at 3 pm or Standup at 5 pm"
      assert String.ends_with?(text, "?")
    end

    test "no event identified on a calendar with one event: the plan asks about that one" do
      {_plan, _specified, text} =
        respond({:event_not_identified, [%{event: "Dentist", date: "", time: "3 pm"}]}, :local_scheduler, "calendar.cancel")

      assert text =~ "Dentist at 3 pm?"
    end

    test "nothing matched what was described: an action result naming the description, with its day" do
      reason = {:no_matching_items, :calendar, %{event: "dentist", date: "friday", time: "3 pm"}}

      {plan, specified, text} = respond(reason, :local_scheduler, "calendar.cancel")

      assert [%Primitive{type: :content, variant: :action_result, content: content}] = plan
      assert content.reply == :no_matching_items
      assert content.result == :no_matching_items
      assert content.criteria == %{event: "dentist", date: "friday", time: "3 pm"}
      assert [%Primitive{content: %{action: "calendar.cancel", content_complete: true}}] = specified

      assert text =~ "dentist on friday at 3 pm"
    end

    test "nothing of the kind at all: the reply says there are none" do
      {plan, _specified, text} = respond({:no_matching_items, :alarm, %{}}, :local_scheduler, "alarm.cancel")

      assert [%Primitive{type: :content, variant: :action_result, content: %{reply: :no_items}}] = plan
      assert text =~ "alarms"
    end

    test "no ringing alarm: the reply says no alarm is ringing" do
      {plan, _specified, text} = respond(:no_ringing_alarm, :local_scheduler, "alarm.snooze")

      assert [%Primitive{type: :content, variant: :action_result, content: %{reply: :no_ringing_alarm}}] = plan
      assert text =~ "alarm"
      assert text =~ "snooze"
    end

    test "a snooze with no length said asks how long, with the intent's own prompt for its duration" do
      {plan, _specified, text} = respond(:missing_duration, :local_scheduler, "alarm.snooze")

      assert [%Primitive{type: :follow_up, variant: :clarification, content: content}] = plan
      assert content.reply == :missing_slots
      assert content.missing_slots == ["duration"]
      assert text == Brain.Analysis.SlotDetector.get_clarification_prompt("duration", "alarm.snooze")
    end

    test "a missing location is asked for with the intent's own prompt" do
      {plan, _specified, text} = respond(:missing_location, :weather, "weather.query")

      assert [%Primitive{type: :follow_up, variant: :clarification, content: content}] = plan
      assert content.reply == :missing_slots
      assert content.missing_slots == ["location"]
      assert text == Brain.Analysis.SlotDetector.get_clarification_prompt("location", "weather.query")
    end

    # Every reason a service or the dispatcher returns, with the primitive and
    # reply it is planned with and a fragment of the reply's wording.
    @planned [
      {:location_not_found, :weather, {:follow_up, :clarification}, :location_not_found, "location"},
      {:side_effects_off, :local_scheduler, {:content, :action_result}, :side_effects_off, "side effects"},
      {:rate_limited, :weather, {:content, :action_result}, :rate_limited, "Weather"},
      {:service_unavailable, :weather, {:content, :action_result}, :unavailable, "Weather"},
      {{:connection_error, :timeout}, :home_assistant, {:content, :action_result}, :unavailable, "Home Assistant"},
      {:service_error, :home_assistant, {:content, :action_result}, :service_failed, "Home Assistant"},
      {{:api_error, 500}, :weather, {:content, :action_result}, :service_failed, "Weather"},
      {{:http_error, 500, "oops"}, :home_assistant, {:content, :action_result}, :service_failed, "Home Assistant"},
      {:unexpected_response, :home_assistant, {:content, :action_result}, :service_failed, "Home Assistant"},
      {%Jason.DecodeError{position: 0, data: "x"}, :home_assistant, {:content, :action_result}, :service_failed,
       "Home Assistant"},
      {:entity_not_found, :home_assistant, {:content, :action_result}, :device_not_found, "device"},
      {:missing_credentials, :home_assistant, {:framing, :boundary}, :not_configured, "set up"},
      {:missing_api_key, :weather, {:framing, :boundary}, :not_configured, "set up"},
      {:invalid_api_key, :weather, {:framing, :boundary}, :invalid_credentials, "credentials"},
      {{:http_error, 401, "unauthorized"}, :home_assistant, {:framing, :boundary}, :invalid_credentials, "credentials"},
      {:service_disabled, :local_scheduler, {:framing, :boundary}, :disabled, "Local Scheduler"},
      {{:unsupported_intent, "smarthome.dance"}, :home_assistant, {:framing, :boundary}, :unsupported,
       "Home Assistant"},
      {:unsupported_intent, :local_scheduler, {:framing, :boundary}, :unsupported, "Local Scheduler"}
    ]

    for {reason, service, {type, variant}, reply, fragment} <- @planned do
      @reason reason
      @service service
      @type_variant {type, variant}
      @reply reply
      @fragment fragment

      test "#{inspect(reason)} is planned as a #{type} #{variant} replying #{reply}" do
        {plan, specified, text} = respond(@reason, @service, "some.intent")
        {type, variant} = @type_variant

        assert [%Primitive{type: ^type, variant: ^variant, content: content}] = plan
        assert content.reply == @reply
        assert content.enrichment_error == @reason
        assert [%Primitive{content: %{content_complete: true}}] = specified
        assert text =~ @fragment
        refute text =~ "$"
      end
    end

    test "the error replaces every chunk's plan" do
      model = model_for("cancel my meeting", "calendar.cancel")
      two_chunks = %{model | analyses: model.analyses ++ [%{hd(model.analyses) | chunk_index: 1, text: "thanks"}]}

      assert [%Primitive{type: :follow_up, variant: :clarification}] =
               DiscoursePlanner.plan(
                 two_chunks,
                 failed({:event_not_identified, [%{event: "Dentist", date: "", time: ""}]}, :local_scheduler, "calendar.cancel")
               )
    end

    test "an error with no planned response raises" do
      model = model_for("cancel my meeting", "calendar.cancel")

      assert_raise ArgumentError, ~r/no planned response for the service error :timeout/, fn ->
        DiscoursePlanner.plan(model, failed(:timeout, :weather, "weather.query"))
      end

      assert_raise ArgumentError, ~r/no planned response for the service error \{:ambiguous, :calendar, \[/, fn ->
        DiscoursePlanner.plan(
          model,
          failed({:ambiguous, :calendar, [%{event: "Dentist", time: "3 pm"}]}, :local_scheduler, "calendar.cancel")
        )
      end

      assert_raise ArgumentError, ~r/no planned response for the service error \{:event_not_identified, \[\]\}/, fn ->
        DiscoursePlanner.plan(model, failed({:event_not_identified, []}, :local_scheduler, "calendar.cancel"))
      end
    end

    test "a failed enrichment with no error raises" do
      assert_raise ArgumentError, ~r/carries no :enrichment_error/, fn ->
        DiscoursePlanner.plan(model_for("cancel", "calendar.cancel"),
          unified_context: %{enrichment: %{enrichment_status: :failed}}
        )
      end
    end

    test "an enrichment that did not fail plans as before" do
      opts = [unified_context: %{enrichment: %{enrichment_status: :not_configured, enrichment_error: nil}}]

      refute Enum.any?(
               DiscoursePlanner.plan(model_for("cancel my meeting", "calendar.cancel"), opts),
               &Map.has_key?(&1.content, :enrichment_error)
             )
    end

    # No registered service returns a success with no data
    # (`DiscoursePlanner.empty_result_primitive/1`); one that did is answered
    # with what that means for the intent it served, never with a reply that
    # claims a result.
    test "a success with no data says the service did it, or returned nothing, as the intent writes or reads" do
      empty = fn service, intent ->
        [
          unified_context: %{
            enrichment: %{enrichment_status: :success, enriched_data: %{}, enrichment_service: service, intent: intent}
          }
        ]
      end

      for {service, intent, reply, wording} <- [
            {:local_scheduler, "timer.cancel", :carried_out, ["Local Scheduler carried that out.", "Local Scheduler took care of that."]},
            {:weather, "weather.query", :nothing_reported,
             ["Weather returned nothing for that.", "I asked Weather, but it had nothing to report."]}
          ] do
        model = model_for("do it", intent)
        [analysis] = model.analyses
        opts = empty.(service, intent)

        assert [%Primitive{type: :content, variant: :action_result, content: content}] =
                 plan = DiscoursePlanner.plan(model, opts)

        assert content.reply == reply
        assert content.enrichment_result == :empty

        specified = ContentSpecifier.specify(plan, analysis, opts)
        assert [%Primitive{content: %{content_complete: true}}] = specified

        {:ok, rendered, text, nil} = SurfaceRealizer.realize(specified, [world_id: @world_id, analysis: analysis] ++ opts)
        assert Enum.all?(rendered, &(&1.source == :service_reply))
        assert text in wording
        refute ResponseEvaluator.silence_preferred?(0.0, plan)
      end
    end

    test "a service reply is never silenced, however low it scores" do
      reply = DiscoursePlanner.plan(model_for("snooze", "alarm.snooze"), failed(:no_ringing_alarm, :local_scheduler, "alarm.snooze"))
      ordinary = DiscoursePlanner.plan(model_for("snooze", "alarm.snooze"))

      refute ResponseEvaluator.silence_preferred?(0.0, reply)
      assert ResponseEvaluator.silence_preferred?(0.0, ordinary)
    end
  end
end
