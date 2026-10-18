defmodule Brain.FactDatabase.PolarAlignmentTest do
  @moduledoc """
  `Brain.FactDatabase.Integration.polar_alignment/3`: how known statements
  answer a polar question. A statement confirms or disconfirms the question's
  proposition only when it states it, word for word up to the auxiliary's
  place; anything else answers nothing.
  """
  use ExUnit.Case, async: false

  alias Brain.FactDatabase.Integration
  alias Brain.ML.{POSTagger, Tokenizer}

  # The POS tagger reads WordNet through its lexical features.
  setup_all do
    Brain.TestHelpers.start_pos_tagger!()
  end

  # As the pipeline tags a question (`Pipeline.get_pos_tags/1`).
  defp align(question, statements) do
    {:ok, model} = POSTagger.get_model()
    Integration.polar_alignment(question, question |> Tokenizer.tokenize_words() |> POSTagger.predict(model), statements)
  end

  @capital "The capital of France is Paris"

  test "a statement of the proposition, either way round when both sides name one thing, confirms it" do
    assert align("Is Paris the capital of France?", [@capital]) == %{alignment: :confirming, statement: @capital}

    assert align("Is the capital of France Paris?", [@capital]) == %{alignment: :confirming, statement: @capital}

    assert align("Is Paris the capital of France?", ["Paris is the capital of France"]) ==
             %{alignment: :confirming, statement: "Paris is the capital of France"}
  end

  test "the same words in other roles state another proposition, and answer nothing" do
    assert align("Is France the capital of Paris?", [@capital]) == nil
  end

  test "a statement about another thing answers nothing: it is not read as a no" do
    assert align("Is Berlin the capital of France?", [@capital]) == nil
    assert align("Is Paris the capital of France?", ["The Berlin Wall fell on November 9, 1989"]) == nil
  end

  test "a statement of the proposition with a negation disconfirms it" do
    negated = "Paris is not the capital of Germany"
    assert align("Is Paris the capital of Germany?", [negated]) == %{alignment: :disconfirming, statement: negated}
    assert align("Is Paris the capital of Germany?", ["Paris isn't the capital of Germany"]).alignment == :disconfirming
  end

  test "statements that both confirm and disconfirm decide nothing" do
    assert align("Is Paris the capital of France?", [@capital, "Paris is not the capital of France"]) == nil
  end

  test "a negated question, or one that is not polar, is answered by no statement" do
    assert align("Isn't Paris the capital of France?", [@capital]) == nil
    assert align("What is the capital of France?", [@capital]) == nil
    assert align("Is Paris the capital of France?", []) == nil
  end
end
