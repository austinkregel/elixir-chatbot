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

  test "its evidence reads the speech act from the words, pooled across the intents that declare it",
       %{table: table} do
    assert {:ok, {:expressive, :greeting}, value} = SpeechActTable.read(table.evidence, "hello")
    assert value > 0
    assert {:ok, {:expressive, :farewell}, _} = SpeechActTable.read(table.evidence, "goodbye")
    assert {:ok, {:directive, :request_information}, _} = SpeechActTable.read(table.evidence, "the weather")
  end

  test "words no speech act's rows hold are read as no speech act", %{table: table} do
    assert SpeechActTable.read(table.evidence, "zyxwv") == :none
  end

  test "out-of-fold evidence is folded as the intent evidence folds the rows, each fold fitted without its rows",
       %{table: table} do
    {folds, models} = SpeechActTable.fold_evidence(@rows, @registry, table)

    assert folds == Brain.Analysis.LexicalEvidence.folds(@rows)
    assert Map.keys(models) == Enum.to_list(0..(Brain.Analysis.LexicalEvidence.fold_count() - 1))

    for {k, model} <- models do
      assert model.prior_strength == table.evidence.prior_strength
      assert model.intents == table.evidence.intents

      held = for {{_text, intent}, ^k} <- Enum.zip(@rows, folds), @registry[intent]["speech_act"] != nil, do: 1
      assert model.fitted_rows == table.evidence.fitted_rows - length(held)
    end
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
