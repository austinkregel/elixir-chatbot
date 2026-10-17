defmodule Brain.Analysis.LexicalEvidenceTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.LexicalEvidence

  @rows [
    {"hello there", "smalltalk.greetings.hello"},
    {"hi there", "smalltalk.greetings.hello"},
    {"hello hello", "smalltalk.greetings.hello"},
    {"hey hi", "smalltalk.greetings.hello"},
    {"what is the weather", "weather.query"},
    {"weather in paris", "weather.query"},
    {"is it raining in paris", "weather.query"},
    {"the weather tomorrow", "weather.query"},
    {"play some jazz", "music.play"},
    {"play music", "music.play"},
    {"play the radio", "music.play"},
    {"put on some jazz", "music.play"}
  ]

  setup do
    {:ok, model: LexicalEvidence.fit(@rows, language: "en", prior_strength: 10)}
  end

  test "an utterance's strongest intent is the one its words were trained with", %{model: model} do
    for {text, intent} <- [{"hello", "smalltalk.greetings.hello"}, {"weather paris", "weather.query"}, {"play jazz", "music.play"}] do
      values = LexicalEvidence.features(model, text)
      {_value, top} = values |> Enum.zip(model.intents) |> Enum.max_by(&elem(&1, 0))
      assert top == intent, "#{inspect(text)} read as #{top}"
    end
  end

  test "every value is in (-1, 1), one per intent", %{model: model} do
    values = LexicalEvidence.features(model, "hello what is the weather")
    assert length(values) == length(model.intents)
    assert Enum.all?(values, &(&1 > -1.0 and &1 < 1.0))
  end

  test "an utterance with no known word scores zero for every intent", %{model: model} do
    assert LexicalEvidence.features(model, "zxqv") == List.duplicate(0.0, length(model.intents))
  end

  test "feature names follow the intents", %{model: model} do
    assert LexicalEvidence.names(model) == Enum.map(model.intents, &String.to_atom("lex_" <> &1))
  end

  test "select picks a prior strength by cross-validation and fits on every row" do
    {model, accuracies} = LexicalEvidence.select(@rows, language: "en")

    assert model.fitted_rows == length(@rows)
    assert Map.has_key?(accuracies, model.prior_strength)
    assert Enum.all?(Map.values(accuracies), &(&1 >= 0.0 and &1 <= 1.0))
  end

  test "a row's fold model was fitted without that row", %{model: model} do
    {folds, models} = LexicalEvidence.fold_models(@rows, model)

    for {{text, _intent}, fold} <- Enum.zip(@rows, folds) do
      assert models[fold].fitted_rows < length(@rows)
      refute Enum.any?(fold_rows(@rows, folds, fold), &(&1 == text))
    end
  end

  test "every fold model scores the full model's intents, in its order" do
    # One row of its own: whichever fold holds it out fits without the intent.
    rows = [{"good morning", "smalltalk.greetings.goodmorning"} | @rows]
    model = LexicalEvidence.fit(rows, language: "en", prior_strength: 10)
    {_folds, models} = LexicalEvidence.fold_models(rows, model)

    for {_k, fold_model} <- models do
      assert fold_model.intents == model.intents
      assert length(LexicalEvidence.features(fold_model, "good morning")) == length(model.intents)
    end
  end

  test "fold models must be built from the rows the model was fitted on", %{model: model} do
    assert_raise ArgumentError, ~r/fold models must use the same rows/, fn ->
      LexicalEvidence.fold_models(tl(@rows), model)
    end
  end

  test "an unsupported language cannot be fitted" do
    assert_raise ArgumentError, ~r/"fr" is not supported/, fn ->
      LexicalEvidence.fit(@rows, language: "fr", prior_strength: 10)
    end
  end

  # The texts a fold's model was fitted on.
  defp fold_rows(rows, folds, k) do
    for {{text, _}, fold} <- Enum.zip(rows, folds), fold != k, do: text
  end
end
