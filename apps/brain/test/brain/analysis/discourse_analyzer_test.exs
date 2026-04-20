defmodule Brain.Analysis.DiscourseAnalyzerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.DiscourseAnalyzer
  alias Brain.Analysis.DiscourseResult

  describe "analyze/2 - addressee detection" do
    test "detects direct address to bot" do
      result = DiscourseAnalyzer.analyze("Hey bot, what's the weather?")

      assert %DiscourseResult{
               addressee: :bot,
               direct_address_detected: true
             } = result

      assert "direct_address" in result.indicators
    end

    test "detects second person pronouns as addressing bot in 1-on-1" do
      result = DiscourseAnalyzer.analyze("Can you help me?")

      assert result.addressee == :bot
      assert "second_person_pronoun" in result.indicators
    end

    test "detects imperative as addressing bot in 1-on-1" do
      result = DiscourseAnalyzer.analyze("Turn on the lights")

      assert result.addressee == :bot
      assert "imperative_mood" in result.indicators
    end

    test "detects questions as expecting bot response in 1-on-1" do
      result = DiscourseAnalyzer.analyze("What is the time?")

      assert result.addressee == :bot
      assert "question" in result.indicators
    end

    test "detects modal you patterns" do
      result = DiscourseAnalyzer.analyze("Would you please help me?")

      assert result.addressee == :bot

      has_modal_indicator = "modal_you_request" in result.indicators
      has_second_person = "second_person_pronoun" in result.indicators

      assert has_modal_indicator or has_second_person,
             "Expected modal_you_request or second_person_pronoun indicator, got: #{inspect(result.indicators)}"
    end

    test "handles multiple bot name patterns" do
      for name <- ["companion", "assistant", "ai"] do
        result = DiscourseAnalyzer.analyze("Hey #{name}, hello!")
        assert result.addressee == :bot, "Failed for: #{name}"
      end
    end
  end

  describe "analyze/2 - with custom options" do
    test "uses custom bot names" do
      result =
        DiscourseAnalyzer.analyze("Hey jarvis, what's up?",
          bot_names: ["jarvis"]
        )

      assert result.addressee == :bot
      assert result.direct_address_detected == true
    end

    test "uses custom participants" do
      result =
        DiscourseAnalyzer.analyze("Hello everyone",
          participants: [:user, :bot, :other_user]
        )

      assert :user in result.participants
      assert :bot in result.participants
      assert :other_user in result.participants
    end
  end

  describe "analyze/2 - ambiguous cases" do
    test "self-referential statements may be ambiguous" do
      result = DiscourseAnalyzer.analyze("I am feeling tired today.")

      # User is making a statement about themselves
      # Could be addressing bot or just thinking out loud
      assert result.addressee in [:bot, :ambiguous]
    end

    test "third person references may indicate third party" do
      result = DiscourseAnalyzer.analyze("He said he would come tomorrow.")

      # Talking about someone else
      assert "third_person_reference" in result.indicators
    end
  end

  describe "analyze/2 - confidence" do
    test "high confidence for direct address" do
      result = DiscourseAnalyzer.analyze("Hey bot, help me!")

      assert result.confidence > 0.7
    end

    test "moderate confidence for implicit address" do
      result = DiscourseAnalyzer.analyze("What's the weather?")

      assert result.confidence > 0.4
    end
  end

  describe "debug_analyze/2" do
    test "returns detailed debug information" do
      result = DiscourseAnalyzer.debug_analyze("Can you help me?")

      assert %{
               result: %DiscourseResult{},
               normalized_text: _,
               detected_pronouns: pronouns,
               has_direct_address: _,
               has_imperative: _,
               is_question: true
             } = result

      assert "you" in pronouns.second_person or "me" in pronouns.first_person
    end
  end
end
