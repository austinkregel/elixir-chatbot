defmodule Brain.Analysis.AnaphoraResolverDataTest do
  @moduledoc """
  AnaphoraResolver: which words are references (the part-of-speech tag and the
  closed-class UD features), how a reference is ranked against the entities of
  a chronological history, and the referent it resolves to.

  Reads the part-of-speech model, so the lexicon, the tagger's lexical
  features and the type hierarchy must be running.
  """
  use ExUnit.Case, async: false

  alias Brain.Analysis.AnaphoraResolver

  defp turn(entities), do: %{entities: entities, intent: nil, timestamp: 0}
  defp entity(type, value, confidence), do: %{entity_type: type, value: value, confidence: confidence}

  describe "references/1" do
    test "a third-person personal pronoun or a demonstrative the tagger tags PRON is a reference" do
      assert [{1, "it", "Neut"}] = AnaphoraResolver.references("turn it off")
      assert [{0, "she", "Fem"}] = AnaphoraResolver.references("she said hello")
      assert [{0, "he", "Masc"}] = AnaphoraResolver.references("he is coming")
      assert [{1, "them", "Prs"}] = AnaphoraResolver.references("turn them on")
      assert [{1, "that", "Dem"}] = AnaphoraResolver.references("play that again")
    end

    test "first- and second-person pronouns, and words that are not pronouns, are not references" do
      assert AnaphoraResolver.references("I like coffee") == []
      assert AnaphoraResolver.references("we should go") == []
      assert AnaphoraResolver.references("can you help me") == []
      assert AnaphoraResolver.references("Paris is beautiful") == []
      assert AnaphoraResolver.references("") == []
      assert AnaphoraResolver.references("   ") == []
    end

    test "a reflexive is not a reference" do
      assert %{"PronType" => "Prs", "Person" => "3", "Reflex" => "Yes"} = Brain.Lexicon.ClosedClass.features("itself")
      refute Enum.any?(AnaphoraResolver.references("the lamp turned itself off"), fn {_, token, _} -> token == "itself" end)
    end

    test "the index is the token's among Tokenizer.tokenize/1's tokens" do
      text = "ok, turn it off."
      [{idx, "it", "Neut"}] = AnaphoraResolver.references(text)
      assert Enum.at(Brain.ML.Tokenizer.tokenize_words(text), idx) == "it"
    end
  end

  describe "candidates/1" do
    test "the history is chronological: its last turn is 1 turn ago and the most recent" do
      history = [
        turn([entity("device", "lamp", 0.9)]),
        turn([entity("device", "fan", 0.8)])
      ]

      assert [{fan, recency_fan, 1}, {lamp, recency_lamp, 2}] = AnaphoraResolver.candidates(history)
      assert fan.value == "fan"
      assert lamp.value == "lamp"
      assert recency_fan == 1.0
      assert recency_lamp < recency_fan
    end

    test "a turn whose entities are not a list of entities raises" do
      assert_raise ArgumentError, ~r/:entities must be a list of entities/, fn ->
        AnaphoraResolver.candidates([%{entities: %{"device" => "lamp"}}])
      end

      assert_raise ArgumentError, ~r/needs a string :entity_type, a :value and a numeric :confidence/, fn ->
        AnaphoraResolver.candidates([turn([%{entity_type: "device", value: "lamp"}])])
      end

      assert_raise ArgumentError, ~r/needs a string :entity_type/, fn ->
        AnaphoraResolver.candidates([turn([%{type: "device", value: "lamp", confidence: 0.9}])])
      end
    end
  end

  describe "resolve/2" do
    test "the most recent compatible entity is the referent" do
      history = [
        turn([entity("device", "lamp", 0.9)]),
        turn([entity("device", "fan", 0.9)])
      ]

      assert {:resolved, [{1, "it", referent}]} = AnaphoraResolver.resolve("turn it off", history)
      assert referent.value == "fan"
    end

    test "the referent's confidence is the resolution's score times the entity's own" do
      [{_, recency, 1}] = AnaphoraResolver.candidates([turn([entity("device", "lamp", 0.9)])])
      {:resolved, [{_, _, high}]} = AnaphoraResolver.resolve("turn it off", [turn([entity("device", "lamp", 0.9)])])
      {:resolved, [{_, _, low}]} = AnaphoraResolver.resolve("turn it off", [turn([entity("device", "lamp", 0.3)])])

      assert high == %{entity_type: "device", value: "lamp", confidence: high.confidence, source: :anaphora_resolution}
      # The same resolution, so the same score: the confidences differ by the
      # entities' own.
      assert_in_delta high.confidence / 0.9, low.confidence / 0.3, 1.0e-9
      assert high.confidence <= 0.9
      assert recency == 1.0

      # A referent from an older turn resolves with a lower score, so with a
      # lower confidence than the same entity's from the last turn.
      {:resolved, [{_, _, older}]} =
        AnaphoraResolver.resolve("turn it off", [turn([entity("device", "lamp", 0.9)]), turn([])])

      assert older.value == "lamp"
      assert older.confidence < high.confidence
    end

    test "nothing to resolve to, or no reference, is no anaphora" do
      assert AnaphoraResolver.resolve("turn it off", []) == {:no_anaphora, []}
      assert AnaphoraResolver.resolve("I like coffee", [turn([entity("device", "lamp", 0.9)])]) == {:no_anaphora, []}
    end
  end

  describe "resolve_and_substitute/2" do
    test "replaces each resolved reference with its referent's value and leaves the rest of the text as it is" do
      history = [turn([entity("device", "lamp", 0.9)])]

      assert {:ok, "ok, turn lamp off.", [%{value: "lamp", source: :anaphora_resolution}]} =
               AnaphoraResolver.resolve_and_substitute("ok, turn it off.", history)
    end

    test "text with nothing resolved comes back unchanged" do
      assert AnaphoraResolver.resolve_and_substitute("hello world", []) == {:ok, "hello world", []}
      assert AnaphoraResolver.resolve_and_substitute("日本語テスト", []) == {:ok, "日本語テスト", []}
    end
  end
end
