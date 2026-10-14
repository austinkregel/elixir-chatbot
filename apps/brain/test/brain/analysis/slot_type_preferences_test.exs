defmodule Brain.Analysis.SlotTypePreferencesTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.SlotTypePreferences

  # A hierarchy of its own, so these test the estimator and not
  # entity_types.json: room, address and city are locations; lights are a
  # device.
  defmodule Hierarchy do
    @parents %{"room" => ["location"], "address" => ["location"], "city" => ["location"], "lights" => ["device"]}

    def all_types, do: ~w(location room address city device lights)
    def parents_of(type), do: Map.get(@parents, type, [])
  end

  @tags ["ADP", "DET", "NOUN", "VERB", :sentence_start]

  @opts [
    language: "en",
    canonical: %{"place_name" => "address"},
    hierarchy: Hierarchy,
    listed: ~w(location room device place_name),
    reading_types: ~w(city lights),
    tag_vocabulary: @tags
  ]

  @flat %{
    global: 6,
    domain: 1,
    intent: 1,
    hierarchy: 0.0,
    tag_global: 5,
    tag_type: 2,
    ordinary_global: 2,
    ordinary_domain: 1,
    ordinary_intent: 1
  }

  defp example(key, intent, spans, ordinary \\ [], gazetteer \\ []) do
    %{
      key: key,
      intent: intent,
      spans: Enum.map(spans, fn {type, tag} -> %{type: type, preceding_tag: tag} end),
      ordinary: ordinary,
      gazetteer: gazetteer
    }
  end

  # Weather asks for addresses; smart-home commands name a room and a device
  # equally often. Two of every three gazetteer matches in a weather request
  # are ordinary words; every gazetteer match in a smart-home command is an
  # entity. Expected values are worked out by hand in the tests.
  defp examples do
    weather =
      for i <- 1..4,
          do: example({:weather, i}, "weather.query", [{"address", "ADP"}], ["NOUN"], [:entity, :ordinary, :ordinary])

    smarthome =
      for i <- 1..4,
          do:
            example({:smarthome, i}, "smarthome.device.switch.on", [{"room", "DET"}, {"device", "DET"}], ["VERB"], [
              :entity,
              :entity
            ])

    weather ++ smarthome
  end

  defp fit(smoothing), do: SlotTypePreferences.fit(examples(), Keyword.put(@opts, :smoothing, smoothing))

  describe "fit/2: the label universe" do
    test "holds the hierarchy, the gold labels and the listed types' labels, with their parents" do
      model = fit(@flat)

      assert model.universe == ~w(address city device lights location room)
      assert model.parents["room"] == ["location"]
      assert model.children["location"] == ~w(address city room)
      assert MapSet.equal?(model.descendants["location"], MapSet.new(~w(location room address city)))
    end

    test "every type the gazetteer reads joins it, as a root when the hierarchy has no parents for it" do
      model = SlotTypePreferences.fit(examples(), @opts |> Keyword.put(:reading_types, ~w(emoji city)) |> Keyword.put(:smoothing, @flat))

      assert "emoji" in model.universe
      assert model.parents["emoji"] == []
    end

    test "the gazetteer's types are required" do
      for bad <- [[], nil, [""]] do
        assert_raise ArgumentError, ~r/:reading_types must list/, fn ->
          SlotTypePreferences.fit(examples(), @opts |> Keyword.put(:reading_types, bad) |> Keyword.put(:smoothing, @flat))
        end
      end
    end

    test "a hierarchy that holds no types is an error, not an empty hierarchy" do
      defmodule Empty do
        def all_types, do: []
        def parents_of(_type), do: []
      end

      assert_raise ArgumentError, ~r/holds no types/, fn ->
        SlotTypePreferences.fit(examples(), @opts |> Keyword.put(:hierarchy, Empty) |> Keyword.put(:smoothing, @flat))
      end
    end
  end

  describe "slot distribution" do
    test "is smoothed from the intent to its domain to the whole data to uniform" do
      dist = SlotTypePreferences.distribution(fit(@flat), {:intent, "weather.query"})

      # P0 = (c + 6/6) / (12 + 6); Pd = (c + P0) / (4 + 1); Pi = (c + Pd) / (4 + 1).
      assert_in_delta dist["address"], 437 / 450, 1.0e-12
      assert_in_delta dist["room"], 5 / 450, 1.0e-12
      assert_in_delta dist["device"], 5 / 450, 1.0e-12
      assert_in_delta dist["location"], 1 / 450, 1.0e-12
      assert_in_delta dist |> Map.values() |> Enum.sum(), 1.0, 1.0e-12
    end

    test "an intent the data never shows backs off to its domain" do
      model = fit(@flat)

      assert SlotTypePreferences.distribution(model, {:intent, "weather.outfit"}) ==
               SlotTypePreferences.distribution(model, {:domain, "weather"})

      assert_in_delta SlotTypePreferences.distribution(model, {:domain, "weather"})["address"], 77 / 90, 1.0e-12
    end

    test "through the hierarchy a child type receives a share of its family's mass" do
      # Room and device are equally common overall and never seen with weather.
      flat = SlotTypePreferences.distribution(fit(@flat), {:intent, "weather.query"})
      assert_in_delta flat["room"], flat["device"], 1.0e-12

      # With half the mass shared, room inherits a third of location's subtree
      # (444/450 over three children); device's family has only itself and lights.
      shared = SlotTypePreferences.distribution(fit(%{@flat | hierarchy: 0.5}), {:intent, "weather.query"})
      assert shared["room"] > shared["device"]
      assert_in_delta shared |> Map.values() |> Enum.sum(), 1.0, 1.0e-12
    end

    test "a context that is neither an intent of a known domain nor a domain raises" do
      model = fit(@flat)
      assert_raise ArgumentError, ~r/not a domain/, fn -> SlotTypePreferences.distribution(model, {:domain, "nowhere"}) end
      assert_raise ArgumentError, ~r/no domain declared/, fn -> SlotTypePreferences.distribution(model, {:intent, "zzz.query"}) end
    end
  end

  describe "weights/3" do
    test "each gold label's probability goes to the most specific listed type that agrees with it" do
      model = fit(@flat)

      # Only location and room listed: an address is read best as a location.
      weather = SlotTypePreferences.weights(model, {:intent, "weather.query"}, ~w(location room))
      assert_in_delta weather["location"], (1 + 437 + 1) / 450, 1.0e-12
      assert_in_delta weather["room"], 5 / 450, 1.0e-12

      # With place_name listed (the gold label address), the address goes to it.
      with_place = SlotTypePreferences.weights(model, {:intent, "weather.query"}, ~w(location room place_name))
      assert_in_delta with_place["place_name"], 437 / 450, 1.0e-12
      assert_in_delta with_place["location"], 2 / 450, 1.0e-12
    end

    test "the ranking follows the data, not the hierarchy: a room is preferred for smart-home commands" do
      model = fit(@flat)

      weather = SlotTypePreferences.weights(model, {:intent, "weather.query"}, ~w(location room device))
      smarthome = SlotTypePreferences.weights(model, {:intent, "smarthome.device.switch.on"}, ~w(location room device))

      assert weather["location"] > weather["room"]
      assert smarthome["room"] > smarthome["location"]
      assert Enum.all?(Map.values(weather) ++ Map.values(smarthome), &(&1 > 0))
    end

    test "nothing listed weighs nothing; a listed type the model was not trained with raises" do
      model = fit(@flat)
      assert SlotTypePreferences.weights(model, {:intent, "weather.query"}, []) == %{}

      assert_raise ArgumentError, ~r/not in the en model's universe/, fn ->
        SlotTypePreferences.weights(model, {:intent, "weather.query"}, ~w(location song))
      end

      assert_raise RuntimeError, ~r/\["song"\].*mix train_slot_type_preferences/, fn ->
        SlotTypePreferences.check_listed!(model, ~w(location song))
      end

      assert SlotTypePreferences.check_listed!(model, ~w(location place_name)) == :ok
    end
  end

  describe "intent_evidence/3" do
    test "is the log of a label's mass under the intent over its mass across every utterance" do
      model = fit(@flat)
      evidence = SlotTypePreferences.intent_evidence(model, "weather.query", ~w(address room location))

      for_intent = SlotTypePreferences.distribution(model, {:intent, "weather.query"})
      overall = SlotTypePreferences.distribution(model, :global)

      assert_in_delta evidence["address"], :math.log(for_intent["address"] / overall["address"]), 1.0e-12

      # A label's mass includes every label that refines it.
      location = fn d -> d["location"] + d["room"] + d["address"] + d["city"] end
      assert_in_delta evidence["location"], :math.log(location.(for_intent) / location.(overall)), 1.0e-12

      # Weather names addresses more often than utterances do overall, rooms less.
      assert evidence["address"] > 0
      assert evidence["room"] < 0
    end

    test "the global distribution sums to one and is the same for every intent" do
      overall = SlotTypePreferences.distribution(fit(@flat), :global)
      assert_in_delta overall |> Map.values() |> Enum.sum(), 1.0, 1.0e-12
      # Room and address are each seen 4 times of 12, device 4 times.
      assert_in_delta overall["room"], overall["address"], 1.0e-12
    end

    test "a label outside the universe raises, as does an intent with no domain" do
      model = fit(@flat)

      assert_raise ArgumentError, ~r/"song" is not in the en model's universe/, fn ->
        SlotTypePreferences.intent_evidence(model, "weather.query", ["song"])
      end

      assert_raise ArgumentError, ~r/no domain declared/, fn ->
        SlotTypePreferences.intent_evidence(model, "zzz.query", ["room"])
      end
    end
  end

  describe "ordinary_rate/2" do
    test "is smoothed from the intent to its domain to every utterance to an even split" do
      model = fit(@flat)

      # 8 of weather's 12 gazetteer spans are ordinary, none of smart home's 8.
      # O0 = (8 + 2/2) / (20 + 2); Od = (8 + O0) / (12 + 1); Oi = (8 + Od) / (12 + 1).
      assert_in_delta SlotTypePreferences.ordinary_rate(model, :global), 9 / 22, 1.0e-12
      assert_in_delta SlotTypePreferences.ordinary_rate(model, {:domain, "weather"}), 185 / 286, 1.0e-12
      assert_in_delta SlotTypePreferences.ordinary_rate(model, {:intent, "weather.query"}), 2473 / 3718, 1.0e-12

      # Od = (0 + O0) / (8 + 1); Oi = (0 + Od) / (8 + 1).
      assert_in_delta SlotTypePreferences.ordinary_rate(model, {:intent, "smarthome.device.switch.on"}), 1 / 198, 1.0e-12
    end

    test "an intent the data never shows takes its domain's rate" do
      model = fit(@flat)

      assert SlotTypePreferences.ordinary_rate(model, {:intent, "weather.outfit"}) ==
               SlotTypePreferences.ordinary_rate(model, {:domain, "weather"})
    end

    test "only the gazetteer spans of rows joined to an intent are counted" do
      unjoined = example(:unjoined, nil, [{"room", "DET"}], [], [:ordinary, :ordinary, :ordinary])
      model = SlotTypePreferences.fit([unjoined | examples()], Keyword.put(@opts, :smoothing, @flat))

      assert model.ordinary_counts.global == {8, 20}
      assert_in_delta SlotTypePreferences.ordinary_rate(model, :global), 9 / 22, 1.0e-12
    end

    test "a context that is neither an intent of a known domain nor a domain raises" do
      model = fit(@flat)
      assert_raise ArgumentError, ~r/not a domain/, fn -> SlotTypePreferences.ordinary_rate(model, {:domain, "nowhere"}) end
      assert_raise ArgumentError, ~r/no domain declared/, fn -> SlotTypePreferences.ordinary_rate(model, {:intent, "zzz.query"}) end
    end
  end

  describe "unlisted_rate/3" do
    test "is the mass of the labels that agree with no listed type" do
      model = fit(@flat)
      dist = SlotTypePreferences.distribution(model, {:intent, "weather.query"})

      # location and room cover location, room, address and city.
      assert_in_delta SlotTypePreferences.unlisted_rate(model, {:intent, "weather.query"}, ~w(location room)),
                      dist["device"] + dist["lights"],
                      1.0e-12

      # A listed type's gold label counts: place_name is an address.
      assert_in_delta SlotTypePreferences.unlisted_rate(model, {:intent, "weather.query"}, ~w(place_name)),
                      1 - dist["address"],
                      1.0e-12
    end

    test "is every span with nothing listed, and none with every root listed" do
      model = fit(@flat)
      assert_in_delta SlotTypePreferences.unlisted_rate(model, {:intent, "weather.query"}, []), 1.0, 1.0e-12
      assert_in_delta SlotTypePreferences.unlisted_rate(model, {:intent, "weather.query"}, ~w(location device)), 0.0, 1.0e-12
    end

    test "weather's spans agree with its listed location far more often than smart home's do" do
      model = fit(@flat)

      assert SlotTypePreferences.unlisted_rate(model, {:intent, "weather.query"}, ~w(location)) <
               SlotTypePreferences.unlisted_rate(model, {:intent, "smarthome.device.switch.on"}, ~w(location))
    end
  end

  describe "reading_likelihoods/2" do
    test "is each reading's probability in the context over its probability across every utterance" do
      model = fit(@flat)
      table = SlotTypePreferences.reading_likelihoods(model, {:intent, "weather.query"})

      o_i = SlotTypePreferences.ordinary_rate(model, {:intent, "weather.query"})
      o_0 = SlotTypePreferences.ordinary_rate(model, :global)
      for_intent = SlotTypePreferences.distribution(model, {:intent, "weather.query"})
      overall = SlotTypePreferences.distribution(model, :global)

      assert_in_delta SlotTypePreferences.reading_likelihood(table, :ordinary), o_i / o_0, 1.0e-12

      # place_name is read through its gold label, address.
      assert_in_delta SlotTypePreferences.reading_likelihood(table, "place_name"),
                      (1 - o_i) * for_intent["address"] / ((1 - o_0) * overall["address"]),
                      1.0e-12

      # A label's mass includes every label that refines it.
      location = fn d -> d["location"] + d["room"] + d["address"] + d["city"] end

      assert_in_delta SlotTypePreferences.reading_likelihood(table, "location"),
                      (1 - o_i) * location.(for_intent) / ((1 - o_0) * location.(overall)),
                      1.0e-12
    end

    test "moves each reading the way the context's data does, listed or not" do
      model = fit(@flat)
      weather = SlotTypePreferences.reading_likelihoods(model, {:intent, "weather.query"})
      smarthome = SlotTypePreferences.reading_likelihoods(model, {:intent, "smarthome.device.switch.on"})

      # Weather names addresses and rarely rooms; a gazetteer match in a weather
      # request is more often an ordinary word than one anywhere.
      assert SlotTypePreferences.reading_likelihood(weather, "address") > 1
      assert SlotTypePreferences.reading_likelihood(weather, "room") < 1
      assert SlotTypePreferences.reading_likelihood(weather, :ordinary) > 1

      # Smart-home commands name rooms and devices, and their gazetteer
      # matches are entities.
      assert SlotTypePreferences.reading_likelihood(smarthome, "room") > 1
      assert SlotTypePreferences.reading_likelihood(smarthome, :ordinary) < 1
    end

    test "every utterance together says nothing about any reading" do
      table = SlotTypePreferences.reading_likelihoods(fit(@flat), :global)

      assert_in_delta SlotTypePreferences.reading_likelihood(table, :ordinary), 1.0, 1.0e-12

      for type <- ~w(address room device lights location city place_name) do
        assert_in_delta SlotTypePreferences.reading_likelihood(table, type), 1.0, 1.0e-12
      end
    end

    test "a type outside the universe raises rather than being weighed as anything" do
      table = SlotTypePreferences.reading_likelihoods(fit(@flat), {:intent, "weather.query"})

      assert_raise ArgumentError, ~r/"emoji" \(gold label "emoji"\) is not in the en model's universe.*mix train_slot_type_preferences/, fn ->
        SlotTypePreferences.reading_likelihood(table, "emoji")
      end
    end
  end

  describe "preceding-tag likelihood" do
    test "is learned per type and for an ordinary word" do
      model = fit(@flat)

      assert SlotTypePreferences.preceding_likelihood(model, "place_name", "ADP") >
               SlotTypePreferences.preceding_likelihood(model, :ordinary, "ADP")

      assert SlotTypePreferences.preceding_likelihood(model, "room", "DET") >
               SlotTypePreferences.preceding_likelihood(model, "room", "ADP")
    end

    test "is smoothed toward every observation: Pg(t) = (c(t) + kg / |T|) / (N + kg)" do
      model = fit(@flat)
      # 4 ADP, 8 DET, 4 NOUN, 4 VERB: 20 observations, 5 tags.
      assert_in_delta model.tag_likelihoods.global["ADP"], (4 + 5 / 5) / (20 + 5), 1.0e-12
      assert_in_delta model.tag_likelihoods.global[:sentence_start], (0 + 5 / 5) / (20 + 5), 1.0e-12
    end

    test "a parent pools its children, and a child with no observations takes its parent's" do
      model = fit(@flat)

      # location pools address (ADP) and room (DET).
      location = model.tag_likelihoods.labels["location"]
      assert location["ADP"] > model.tag_likelihoods.global["ADP"]
      assert location["DET"] > model.tag_likelihoods.global["DET"]

      for tag <- @tags do
        assert_in_delta SlotTypePreferences.preceding_likelihood(model, "city", tag), location[tag], 1.0e-12
      end
    end

    test "a type outside the universe has no counts and no parents, so it takes the global distribution" do
      model = fit(@flat)
      assert SlotTypePreferences.preceding_likelihood(model, "emoji", "ADP") == model.tag_likelihoods.global["ADP"]
    end

    test "a tag outside the vocabulary raises" do
      assert_raise ArgumentError, ~r/not a tag the model knows/, fn ->
        SlotTypePreferences.preceding_likelihood(fit(@flat), "room", "INTJ")
      end
    end

    test "an observed tag outside the vocabulary fails the fit" do
      bad = [example(:bad, "weather.query", [{"address", "INTJ"}])]

      assert_raise ArgumentError, ~r/"INTJ" in :bad is not in the tag vocabulary/, fn ->
        SlotTypePreferences.fit(bad, Keyword.put(@opts, :smoothing, @flat))
      end
    end
  end

  describe "select/2" do
    # Two weather intents: in `distinct` each always asks for its own label, in
    # `shared` both ask for the same mixture.
    defp weather_rows(distinct?) do
      for i <- 1..60 do
        intent = if rem(i, 2) == 0, do: "weather.query", else: "weather.outfit"

        label =
          cond do
            distinct? and intent == "weather.query" -> "address"
            distinct? -> "city"
            rem(div(i, 2), 2) == 0 -> "address"
            true -> "city"
          end

        # In `distinct` a query's gazetteer matches are ordinary words and an
        # outfit request's are entities; in `shared` each intent's rows
        # alternate between the two, as their labels do.
        gazetteer =
          cond do
            distinct? and intent == "weather.query" -> [:ordinary, :ordinary]
            distinct? -> [:entity, :entity]
            rem(div(i, 2), 2) == 0 -> [:ordinary, :ordinary]
            true -> [:entity, :entity]
          end

        example({:row, i}, intent, [{label, if(label == "address", do: "ADP", else: "DET")}], ["NOUN", "VERB"], gazetteer)
      end
    end

    test "chooses each smoothing value as the best cross-validated one of its grid, in order" do
      {model, report} = SlotTypePreferences.select(examples() ++ weather_rows(true), @opts)

      assert SlotTypePreferences.smoothing_keys() ==
               ~w(global domain intent hierarchy tag_global tag_type ordinary_global ordinary_domain ordinary_intent)a

      for key <- SlotTypePreferences.smoothing_keys() do
        scores = Map.fetch!(report, key)
        best = scores |> Enum.map(&elem(&1, 1)) |> Enum.max()
        {first_best, _} = Enum.find(scores, fn {_value, score} -> score == best end)
        assert report.chosen[key] == first_best, "#{key}: #{inspect(scores)}"
      end

      assert model.smoothing == report.chosen
      assert model.cross_validation == report
      assert report.observations.slots == 8 + 4 + 60
      assert report.observations.gazetteer_spans == 12 + 8 + 120
    end

    test "intents that differ keep their own counts; intents that agree lean on their domain" do
      {distinct, _} = SlotTypePreferences.select(weather_rows(true), @opts)
      {shared, _} = SlotTypePreferences.select(weather_rows(false), @opts)

      assert distinct.smoothing.intent < shared.smoothing.intent
      assert distinct.smoothing.ordinary_intent < shared.smoothing.ordinary_intent
    end

    test "the ordinary-word rate it chooses follows each intent's data" do
      {model, _} = SlotTypePreferences.select(weather_rows(true), @opts)

      assert SlotTypePreferences.ordinary_rate(model, {:intent, "weather.query"}) > 0.9
      assert SlotTypePreferences.ordinary_rate(model, {:intent, "weather.outfit"}) < 0.1
    end

    test "rows joined to an intent with no gazetteer span cannot choose the ordinary-word smoothing" do
      rows = Enum.map(weather_rows(true), &%{&1 | gazetteer: []})

      assert_raise ArgumentError, ~r/no example joined to an intent has a gazetteer span/, fn ->
        SlotTypePreferences.select(rows, @opts)
      end
    end

    test "the smoothing is not the caller's to pass" do
      assert_raise ArgumentError, ~r/chooses the smoothing/, fn ->
        SlotTypePreferences.select(examples(), Keyword.put(@opts, :smoothing, @flat))
      end
    end
  end

  describe "example!/4" do
    @row %{
      "text" => "turn on the kitchen lights",
      "expected" => [%{"type" => "room", "value" => "kitchen"}, %{"type" => "device", "value" => "lights"}]
    }

    defp tagger, do: fn ["turn", "on", "the", "kitchen", "lights"] -> ["VERB", "ADP", "DET", "NOUN", "NOUN"] end

    test "records each span's gold label with the tag before it, the tag before every other token, and each gazetteer span's outcome" do
      # The gazetteer reads "turn on" (no gold span overlaps it), "kitchen"
      # (a gold span) and "the kitchen" (overlapping one).
      gazetteer = fn ["turn", "on", "the", "kitchen", "lights"] -> [{0, 1}, {3, 3}, {2, 3}] end

      assert SlotTypePreferences.example!(@row, "smarthome.device.switch.on", tagger(), gazetteer) == %{
               key: "turn on the kitchen lights",
               intent: "smarthome.device.switch.on",
               spans: [%{type: "room", preceding_tag: "DET"}, %{type: "device", preceding_tag: "NOUN"}],
               ordinary: [:sentence_start, "VERB", "ADP"],
               gazetteer: [:ordinary, :entity, :entity]
             }
    end

    test "a gazetteer span outside the row's tokens raises" do
      assert_raise ArgumentError, ~r/not a \{first, last\} range of its 5 tokens/, fn ->
        SlotTypePreferences.example!(@row, nil, tagger(), fn _ -> [{4, 5}] end)
      end
    end

    test "a tagger that does not tag every token raises" do
      row = %{"text" => "play jazz", "expected" => [%{"type" => "genre", "value" => "jazz"}]}

      assert_raise RuntimeError, ~r/1 tags for 2 tokens/, fn ->
        SlotTypePreferences.example!(row, nil, fn _ -> ["VERB"] end, fn _ -> [] end)
      end
    end
  end

  describe "intents_by_text!/1" do
    test "joins each text to its intent, and refuses a text with two" do
      assert SlotTypePreferences.intents_by_text!([%{"text" => "hi", "intent" => "smalltalk.greet"}]) == %{"hi" => "smalltalk.greet"}

      assert_raise ArgumentError, ~r/labeled both smalltalk.greet and smalltalk.bye/, fn ->
        SlotTypePreferences.intents_by_text!([
          %{"text" => "hi", "intent" => "smalltalk.greet"},
          %{"text" => "hi", "intent" => "smalltalk.bye"}
        ])
      end
    end
  end
end
