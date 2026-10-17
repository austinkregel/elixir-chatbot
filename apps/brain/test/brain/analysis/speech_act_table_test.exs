defmodule Brain.Analysis.SpeechActTableTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.SpeechActTable

  @registry %{
    "smalltalk.greetings.hello" => %{"category" => "expressive", "speech_act" => "greeting"},
    "smalltalk.greetings.bye" => %{"category" => "expressive", "speech_act" => "farewell"},
    "weather.query" => %{"category" => "directive", "speech_act" => "request_information"},
    # Declares no sub-type: it is learned from its utterances.
    "smalltalk.greetings.whatsup" => %{"category" => "expressive"}
  }

  @rows [
    {"hello there", "smalltalk.greetings.hello"},
    {"hi hello", "smalltalk.greetings.hello"},
    {"hey hello there", "smalltalk.greetings.hello"},
    {"goodbye then", "smalltalk.greetings.bye"},
    {"bye bye", "smalltalk.greetings.bye"},
    {"see you goodbye", "smalltalk.greetings.bye"},
    {"what is the weather", "weather.query"},
    {"weather in paris", "weather.query"},
    {"is it raining", "weather.query"},
    {"hello whats up", "smalltalk.greetings.whatsup"},
    {"hey hello", "smalltalk.greetings.whatsup"}
  ]

  setup do
    {:ok, table: SpeechActTable.build(@rows, @registry, "en")}
  end

  test "a declared sub-type is used as declared", %{table: table} do
    assert SpeechActTable.lookup!(table, "smalltalk.greetings.hello") == {:expressive, :greeting}
    assert SpeechActTable.lookup!(table, "weather.query") == {:directive, :request_information}
    assert table.entries["weather.query"].sub_type_source == :declared
  end

  test "an undeclared sub-type is the one its utterances read as", %{table: table} do
    assert SpeechActTable.lookup!(table, "smalltalk.greetings.whatsup") == {:expressive, :greeting}
    assert table.entries["smalltalk.greetings.whatsup"].sub_type_source == :learned
  end

  test "an intent the table does not hold raises", %{table: table} do
    assert_raise RuntimeError, ~r/no en speech act for intent "music.play"/, fn ->
      SpeechActTable.lookup!(table, "music.play")
    end
  end

  test "a training intent missing from the registry raises" do
    assert_raise RuntimeError, ~r/"music.play" is not in the intent registry/, fn ->
      SpeechActTable.build([{"play jazz", "music.play"} | @rows], @registry, "en")
    end
  end
end
