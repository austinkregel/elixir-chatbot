defmodule Brain.ML.TokenizerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Tokenizer

  describe "tokenize/1" do
    test "tokenizes simple text" do
      tokens = Tokenizer.tokenize("hello world")

      assert length(tokens) == 2
      assert Enum.at(tokens, 0).text == "hello"
      assert Enum.at(tokens, 1).text == "world"
    end

    test "tracks token positions" do
      tokens = Tokenizer.tokenize("hello world")

      first = Enum.at(tokens, 0)
      assert first.start_pos == 0
      assert first.end_pos == 4

      second = Enum.at(tokens, 1)
      assert second.start_pos == 6
      assert second.end_pos == 10
    end

    test "handles punctuation" do
      tokens = Tokenizer.tokenize("Hello, world!")
      texts = Enum.map(tokens, & &1.text)
      assert "Hello" in texts
      assert "," in texts
      assert "world" in texts
      assert "!" in texts
    end

    test "handles contractions" do
      tokens = Tokenizer.tokenize("I'm going to don't")

      texts = Enum.map(tokens, & &1.text)
      assert length(texts) >= 4
      assert "going" in texts
      assert "to" in texts
      has_contraction = "I'm" in texts or ("I" in texts and "m" in texts)
      assert has_contraction
    end

    test "handles numbers" do
      tokens = Tokenizer.tokenize("I have 42 apples")

      number_token = Enum.find(tokens, &(&1.type == :number))
      assert number_token != nil
      assert number_token.text == "42"
    end

    test "handles empty string" do
      tokens = Tokenizer.tokenize("")
      assert tokens == []
    end

    test "handles unicode characters" do
      tokens = Tokenizer.tokenize("Café résumé naïve")

      texts = Enum.map(tokens, & &1.text)
      assert "Café" in texts
      assert "résumé" in texts
      assert "naïve" in texts
    end

    test "handles compound words with hyphens" do
      tokens = Tokenizer.tokenize("well-known self-driving")

      texts = Enum.map(tokens, & &1.text)
      assert "well-known" in texts
      assert "self-driving" in texts
    end
  end

  describe "tokenize_words/1" do
    test "returns just text values" do
      words = Tokenizer.tokenize_words("hello world")
      assert words == ["hello", "world"]
    end
  end

  describe "tokenize_normalized/1" do
    test "returns lowercase normalized tokens" do
      words = Tokenizer.tokenize_normalized("Hello WORLD")
      assert words == ["hello", "world"]
    end

    test "filters short tokens with min_length option" do
      words = Tokenizer.tokenize_normalized("I am a test", min_length: 2)
      refute "I" in words
      refute "a" in words
      assert "am" in words
      assert "test" in words
    end

    test "includes numbers by default" do
      words = Tokenizer.tokenize_normalized("I have 5 items")
      assert "5" in words
    end

    test "can exclude numbers" do
      words = Tokenizer.tokenize_normalized("I have 5 items", include_numbers: false)
      refute "5" in words
    end
  end

  describe "normalize/1" do
    test "lowercases and trims text" do
      assert Tokenizer.normalize("  HELLO World  ") == "hello world"
    end

    test "collapses multiple spaces" do
      assert Tokenizer.normalize("hello    world") == "hello world"
    end
  end

  describe "split_sentences/1" do
    test "splits on sentence boundaries" do
      sentences = Tokenizer.split_sentences("Hello world. How are you? I'm fine!")

      assert length(sentences) == 3
      assert Enum.at(sentences, 0).text =~ "Hello world."
      assert Enum.at(sentences, 1).text =~ "How are you?"
      assert Enum.at(sentences, 2).text =~ "I'm fine!"
    end

    test "handles empty string" do
      sentences = Tokenizer.split_sentences("")
      assert sentences == []
    end
  end

  describe "helper functions" do
    test "word_char? identifies letters and digits" do
      assert Tokenizer.word_char?("a") == true
      assert Tokenizer.word_char?("Z") == true
      assert Tokenizer.word_char?("5") == true
      assert Tokenizer.word_char?(" ") == false
      assert Tokenizer.word_char?(",") == false
    end

    test "whitespace? identifies whitespace" do
      assert Tokenizer.whitespace?(" ") == true
      assert Tokenizer.whitespace?("\t") == true
      assert Tokenizer.whitespace?("\n") == true
      assert Tokenizer.whitespace?("a") == false
    end

    test "punctuation? identifies punctuation" do
      assert Tokenizer.punctuation?(",") == true
      assert Tokenizer.punctuation?(".") == true
      assert Tokenizer.punctuation?("!") == true
      assert Tokenizer.punctuation?("a") == false
    end
  end

  describe "extract_numbers/1" do
    test "extracts numbers with positions" do
      numbers = Tokenizer.extract_numbers("I have 42 apples and 3 oranges")

      assert length(numbers) == 2
      {num1, _, _} = Enum.at(numbers, 0)
      {num2, _, _} = Enum.at(numbers, 1)
      assert num1 == "42"
      assert num2 == "3"
    end

    test "handles decimals" do
      numbers = Tokenizer.extract_numbers("The price is 19.99")

      assert length(numbers) == 1
      {num, _, _} = Enum.at(numbers, 0)
      assert num == "19.99"
    end
  end

  describe "extract_dates/1" do
    test "extracts relative dates" do
      tokens = Tokenizer.tokenize("I'll do it tomorrow")
      dates = Tokenizer.extract_dates(tokens)

      assert length(dates) == 1
      {type, value, _, _} = Enum.at(dates, 0)
      assert type == :relative_date
      assert value == "tomorrow"
    end

    test "extracts day names" do
      tokens = Tokenizer.tokenize("See you on Monday")
      dates = Tokenizer.extract_dates(tokens)

      assert length(dates) == 1
      {type, value, _, _} = Enum.at(dates, 0)
      assert type == :day_name
      assert value == "Monday"
    end

    test "extracts month names" do
      tokens = Tokenizer.tokenize("My birthday is in January")
      dates = Tokenizer.extract_dates(tokens)

      assert dates != []
      types = Enum.map(dates, fn {type, _, _, _} -> type end)
      assert :month_name in types
    end
  end
end