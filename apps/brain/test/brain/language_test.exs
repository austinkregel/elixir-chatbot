defmodule Brain.LanguageTest do
  use ExUnit.Case, async: true

  alias Brain.Language

  describe "resolve!/1" do
    test "a browser that names no language gets English" do
      assert Language.resolve!(nil) == "en"
      assert Language.resolve!("") == "en"
      assert Language.resolve!("*") == "en"
    end

    test "regional variants count as their base language" do
      assert Language.resolve!("en-US") == "en"
      assert Language.resolve!("en-GB,en;q=0.9") == "en"
    end

    test "only the top choice counts" do
      assert Language.resolve!("en-US,fr;q=0.8") == "en"

      assert_raise ArgumentError, ~r/"fr" is not supported/, fn ->
        Language.resolve!("fr-FR,en;q=0.9")
      end
    end

    test "the top choice is the highest quality, not the first listed" do
      assert_raise ArgumentError, ~r/"de" is not supported/, fn ->
        Language.resolve!("en;q=0.5,de;q=0.9")
      end
    end

    test "an unsupported language raises, naming it" do
      assert_raise ArgumentError, ~r/"ja" is not supported\. Supported languages: en/, fn ->
        Language.resolve!("ja")
      end
    end

    test "an unreadable quality raises" do
      assert_raise ArgumentError, ~r/unreadable quality/, fn ->
        Language.resolve!("en;q=high")
      end
    end
  end

  describe "validate!/1" do
    test "accepts a supported code and raises on any other" do
      assert Language.validate!("en") == "en"
      assert_raise ArgumentError, fn -> Language.validate!("es") end
    end
  end
end
