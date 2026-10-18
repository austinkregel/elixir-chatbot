defmodule Brain.Analysis.ActiveContextsTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.ActiveContexts

  @registry %{
    "smarthome.device.volume.up" => %{
      "input_contexts" => [%{"name" => "device_volume", "lifespan" => 2}],
      "output_contexts" => [%{"name" => "device_volume", "lifespan" => 5}]
    },
    "smarthome.device.volume" => %{
      "input_contexts" => [%{"name" => "device_volume", "lifespan" => 2}],
      "output_contexts" => []
    },
    "weather.query" => %{
      "input_contexts" => [%{"name" => "weather", "lifespan" => 2}],
      "output_contexts" => [%{"name" => "weather", "lifespan" => 2}, %{"name" => "room", "lifespan" => 1}]
    },
    "smalltalk.greetings.hello" => %{"input_contexts" => [], "output_contexts" => []}
  }

  setup do
    {:ok, tables: ActiveContexts.from_registry(@registry)}
  end

  test "the vocabulary is every context some intent continues from", %{tables: tables} do
    assert tables.vocabulary == ["device_volume", "weather"]
    assert ActiveContexts.names(tables) == [:ctx_device_volume, :ctx_weather]
  end

  test "a context stays active for its lifespan in turns", %{tables: tables} do
    volume = %{intent: "smarthome.device.volume.up"}
    hello = %{intent: "smalltalk.greetings.hello"}

    assert ActiveContexts.from_history([volume], tables) == ["device_volume"]
    assert ActiveContexts.from_history([volume, hello, hello, hello, hello], tables) == ["device_volume"]
    assert ActiveContexts.from_history([volume, hello, hello, hello, hello, hello], tables) == []
  end

  test "contexts from several earlier turns combine; a short lifespan lapses first", %{tables: tables} do
    history = [%{intent: "smarthome.device.volume.up"}, %{intent: "weather.query"}]
    assert ActiveContexts.from_history(history, tables) == ["device_volume", "room", "weather"]

    later = history ++ [%{intent: "smalltalk.greetings.hello"}]
    assert ActiveContexts.from_history(later, tables) == ["device_volume", "weather"]
  end

  test "an intent with no registry entry sets no context", %{tables: tables} do
    assert ActiveContexts.from_history([%{intent: "question.factual"}, %{intent: nil}], tables) == []
  end

  test "a gold row's follow-up context is the turn's active context" do
    assert ActiveContexts.for_row(%{"text" => "to 5", "followup_context" => "device_volume"}) == ["device_volume"]
    assert ActiveContexts.for_row(%{"text" => "turn it up"}) == []
  end

  test "features mark the active contexts in vocabulary order", %{tables: tables} do
    assert ActiveContexts.features(["weather"], tables) == [0.0, 1.0]
    assert ActiveContexts.features([], tables) == [0.0, 0.0]
  end

  test "a context no intent continues from raises", %{tables: tables} do
    assert ActiveContexts.validate!(["weather", "device_volume"], tables) == ["device_volume", "weather"]

    assert_raise ArgumentError, ~r/"room"/, fn -> ActiveContexts.validate!(["room"], tables) end
  end
end
