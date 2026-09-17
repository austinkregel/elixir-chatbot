defmodule Brain.Lexicon.ContractTest do
  @moduledoc """
  Pins the lexicon's current output so the WordNet -> `Brain.Lexicon` swap can be
  shown not to change behaviour.

  Every value here was measured against WordNet 3.1 as loaded by
  `Brain.ML.Lexicon`, at commit 61b0817. These are characterization tests: they
  assert what the system does today, not what it ideally should do. A failure
  means the swap changed an answer, which is the thing the swap must not do.

  Functions that `Brain.Lexicon` does not expose yet are pinned against
  `Brain.ML.Lexicon` here. When the facade gains them, the same expectations get
  asserted against the facade so both layers are proven to agree.
  """
  use ExUnit.Case, async: false

  alias Brain.Lexicon
  alias Brain.ML.Lexicon, as: WordNet

  describe "known_word?/1" do
    test "recognises words WordNet carries and rejects one it does not" do
      for word <- ~w(dog bank run happy unable quickly) do
        assert WordNet.known_word?(word), "expected WordNet to know #{word}"
      end

      refute WordNet.known_word?("zorbl")
    end
  end

  describe "pos/1" do
    test "returns the POS atoms WordNet assigns" do
      assert Enum.sort(WordNet.pos("dog")) == [:noun, :verb]
      assert Enum.sort(WordNet.pos("bank")) == [:noun, :verb]
      assert Enum.sort(WordNet.pos("run")) == [:noun, :verb]
      assert Enum.sort(WordNet.pos("happy")) == [:adj, :adj_satellite]
      assert Enum.sort(WordNet.pos("quickly")) == [:adv]
      assert Enum.sort(WordNet.pos("unable")) == [:adj, :adj_satellite]
    end

    test "an unknown word has no POS" do
      assert WordNet.pos("zorbl") == []
    end
  end

  describe "lemma/1" do
    test "resolves irregular forms from WordNet's morphological exceptions" do
      assert WordNet.lemma("running") == "run"
      assert WordNet.lemma("ran") == "run"
      assert WordNet.lemma("better") == "good"
      assert WordNet.lemma("geese") == "goose"
    end

    test "leaves regular inflections untouched" do
      # Current behaviour, pinned deliberately. wn_exc.pl holds irregular forms
      # only, so regular plurals are not reduced. `Tokenizer.tokenize_lemmatized/2`
      # documents "cities" -> "city", which does not happen.
      assert WordNet.lemma("cities") == "cities"
      assert WordNet.lemma("dogs") == "dogs"
    end

    test "lookup does not consult lemma, so inflected forms read as unknown" do
      # `lemma/1` resolves "geese" to "goose", but `senses/1` and `known_word?/1`
      # look up the surface form only. So an ordinary plural is OOV, which
      # understates `compute_lexical_coverage/1` for any text containing one.
      assert WordNet.lemma("geese") == "goose"
      assert WordNet.senses("goose") != []

      assert WordNet.senses("geese") == []
      refute WordNet.known_word?("geese")
      assert Lexicon.oov?("geese")

      refute WordNet.known_word?("dogs")
      assert Lexicon.oov?("dogs")
    end
  end

  describe "senses/1" do
    test "returns every sense with synset id, POS and definition" do
      senses = WordNet.senses("dog")
      assert length(senses) == 8

      first = List.first(senses)
      assert first.synset_id == 102_086_723
      assert first.pos == :noun

      assert first.definition ==
               "a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds"
    end

    test "polysemous words return all senses" do
      senses = WordNet.senses("bank")
      assert length(senses) == 18
      assert List.first(senses).synset_id == 109_236_472
    end

    test "an unknown word has no senses" do
      assert WordNet.senses("zorbl") == []
    end
  end

  describe "definition/1" do
    test "returns the gloss of the first sense" do
      assert {:ok, defn} = WordNet.definition("happy")
      assert defn == "enjoying or showing or marked by joy or pleasure"
    end
  end

  describe "synonyms/1,2" do
    test "returns synset members across every sense of the word" do
      # Not sense-filtered: "andiron" (firedog) and "cad" (a despicable person)
      # are both here because they share a synset with some sense of "dog".
      assert WordNet.synonyms("dog") |> Enum.sort() |> Enum.take(5) ==
               ["andiron", "blackguard", "bounder", "cad", "canis familiaris"]

      assert WordNet.synonyms("happy") |> Enum.sort() == ["felicitous", "glad", "well-chosen"]
    end

    test "filters by POS when one is given" do
      assert WordNet.synonyms("run", :verb) |> Enum.sort() |> Enum.take(5) ==
               ["be given", "black market", "bleed", "break away", "bunk"]
    end
  end

  describe "hypernyms/1 and hypernym_chain/3" do
    test "walks one level up the IS-A hierarchy" do
      assert WordNet.hypernyms("dog") |> Enum.sort() |> Enum.take(5) ==
               ["blighter", "bloke", "catch", "chap", "cuss"]
    end

    test "walks the chain to the requested depth" do
      assert WordNet.hypernym_chain("car", :noun, max_depth: 3) ==
               [
                 "motor vehicle",
                 "automotive vehicle",
                 "self-propelled vehicle",
                 "wheeled vehicle"
               ]
    end

    test "a deeper chain keeps its head stable" do
      chain = WordNet.hypernym_chain("dog", :noun, max_depth: 5)
      assert length(chain) == 25

      assert Enum.take(chain, 5) ==
               ["domestic animal", "domesticated animal", "canine", "canid", "animal"]
    end
  end

  describe "antonyms/1" do
    test "returns the antonym lemmas" do
      assert WordNet.antonyms("unable") == ["able"]
      assert WordNet.antonyms("happy") == ["unhappy"]
      assert WordNet.antonyms("hopeless") == ["hopeful"]
    end

    test "antonymy is symmetric for plain opposites" do
      assert WordNet.antonyms("hot") == ["cold"]
      assert WordNet.antonyms("cold") == ["hot"]
    end
  end

  describe "word_similarity/2" do
    test "Wu-Palmer similarity over noun hypernym paths" do
      assert Float.round(WordNet.word_similarity("dog", "cat"), 4) == 0.8235
      assert Float.round(WordNet.word_similarity("dog", "car"), 4) == 0.7111
    end

    test "adjectives have no hypernym path, so similarity is zero" do
      # Pinned as current behaviour: WordNet gives adjectives no IS-A chain, so
      # Wu-Palmer cannot score them even when they are near-synonyms.
      assert WordNet.word_similarity("happy", "glad") == 0.0
    end
  end

  describe "expand_with_synonyms/2" do
    test "adds a synonym only when it is in the vocabulary" do
      # The vocabulary is a map of word => index, matching what
      # Brain.Memory.Embedder and Brain.ML.SimpleClassifier pass.
      vocabulary = %{"canis familiaris" => 0, "automobile" => 1}

      expanded = WordNet.expand_with_synonyms(["dog"], vocabulary)
      assert "dog" in expanded
      assert "canis familiaris" in expanded
    end

    test "leaves a token alone when no synonym is in the vocabulary" do
      assert WordNet.expand_with_synonyms(["dog"], %{"unrelated" => 0}) == ["dog"]
    end

    test "passes through tokens already in the vocabulary" do
      assert WordNet.expand_with_synonyms(["dog"], %{"dog" => 0}) == ["dog"]
    end
  end

  describe "Brain.Lexicon facade" do
    test "primary_domain/1 returns the lexicographer domain of the most-tagged sense" do
      assert Lexicon.primary_domain("dog") == :noun_animal
      assert Lexicon.primary_domain("bank") == :noun_object
      assert Lexicon.primary_domain("run") == :verb_motion
    end

    test "polysemy_count/1 counts senses" do
      assert Lexicon.polysemy_count("dog") == 8
      assert Lexicon.polysemy_count("bank") == 18
      assert Lexicon.polysemy_count("run") == 57
      assert Lexicon.polysemy_count("zorbl") == 0
    end

    test "lookup/2 resolves a known word to the WordNet tier" do
      assert {:wordnet, senses} = Lexicon.lookup("dog")
      assert length(senses) == 8
      assert Enum.all?(senses, &Map.has_key?(&1, :lexical_domain))
    end

    test "lookup/2 reports an unknown word as OOV" do
      assert Lexicon.lookup("zorbl") == :oov
      assert Lexicon.oov?("zorbl")
      assert Lexicon.known?("dog")
    end

    test "synset_ids/2 filters by POS" do
      assert length(Lexicon.synset_ids("dog", :noun)) == 7
      assert length(Lexicon.synset_ids("dog")) == 8
    end

    test "hypernym_depth/1 measures the full chain" do
      assert Lexicon.hypernym_depth("dog") == 46
    end

    test "domain_atoms/0 is the 45-domain lexicographer set" do
      # ChunkFeatures group 10 is sized from this list, so its length is part of
      # the feature vector's shape.
      assert length(Lexicon.domain_atoms()) == 45
    end

    test "conceptnet relations are available for a common concept" do
      assert map_size(Lexicon.conceptnet_relations("dog")) == 21
      assert Lexicon.conceptnet_relation_counts("dog")["IsA"] > 0
    end
  end

  describe "the facade agrees with WordNet while the brain owns no facts" do
    # Every lexical call site is being moved from Brain.ML.Lexicon to
    # Brain.Lexicon. That move is only safe if, with nothing in the owned store,
    # the facade returns exactly what WordNet returns -- same values, same order.
    @words ~w(dog bank run happy glad unable quickly car cat geese goose dogs
              running zorbl cold hot hopeless light fair spring)

    setup do
      assert Brain.Lexicon.UserDefined.fact_count() == 0,
             "the global lexicon store holds facts, so the facade is no longer " <>
               "expected to equal WordNet; this comparison needs an empty store"

      :ok
    end

    test "synonyms/2" do
      for word <- @words, pos <- [nil, :noun, :verb, :adj] do
        assert Lexicon.synonyms(word, pos) == WordNet.synonyms(word, pos),
               "synonyms(#{inspect(word)}, #{inspect(pos)}) diverged"
      end
    end

    test "hypernyms/2" do
      for word <- @words, pos <- [nil, :noun, :verb] do
        assert Lexicon.hypernyms(word, pos) == WordNet.hypernyms(word, pos),
               "hypernyms(#{inspect(word)}, #{inspect(pos)}) diverged"
      end
    end

    test "antonyms/1" do
      for word <- @words do
        assert Lexicon.antonyms(word) == WordNet.antonyms(word),
               "antonyms(#{inspect(word)}) diverged"
      end
    end

    test "definition/2" do
      for word <- @words, pos <- [nil, :noun, :verb] do
        assert Lexicon.definition(word, pos) == WordNet.definition(word, pos),
               "definition(#{inspect(word)}, #{inspect(pos)}) diverged"
      end
    end

    test "hypernym_chain/3" do
      for word <- @words, pos <- [nil, :noun], depth <- [3, 8] do
        assert Lexicon.hypernym_chain(word, pos, max_depth: depth) ==
                 WordNet.hypernym_chain(word, pos, max_depth: depth),
               "hypernym_chain(#{inspect(word)}, #{inspect(pos)}, #{depth}) diverged"
      end
    end

    test "senses/1, pos/1, known_word?/1 and lemma/1" do
      for word <- @words do
        assert Lexicon.senses(word) == WordNet.senses(word), "senses(#{inspect(word)}) diverged"
        assert Lexicon.pos(word) == WordNet.pos(word), "pos(#{inspect(word)}) diverged"

        assert Lexicon.known_word?(word) == WordNet.known_word?(word),
               "known_word?(#{inspect(word)}) diverged"

        assert Lexicon.lemma(word) == WordNet.lemma(word), "lemma(#{inspect(word)}) diverged"
      end
    end

    test "expand_with_synonyms/2" do
      vocabularies = [
        %{},
        %{"canis familiaris" => 0},
        %{"dog" => 0, "goose" => 1},
        %{"run" => 0, "felicitous" => 1, "automobile" => 2}
      ]

      for vocabulary <- vocabularies, tokens <- [@words, ~w(geese running happy car)] do
        assert Lexicon.expand_with_synonyms(tokens, vocabulary) ==
                 WordNet.expand_with_synonyms(tokens, vocabulary),
               "expand_with_synonyms diverged for vocabulary #{inspect(vocabulary)}"
      end
    end
  end

  describe "disambiguate/3" do
    # Shape only. Sense selection is currently not context-sensitive for "bank"
    # (river and money contexts both return synset 109236472), so pinning the
    # selected sense here would lock that in as expected.
    test "returns a well-formed result for a polysemous word" do
      assert {:ok, result} = Lexicon.disambiguate("bank", nil, ~w(river water shore))
      assert is_integer(result.synset_id)
      assert is_atom(result.domain)
      assert result.confidence > 0.0 and result.confidence <= 1.0
    end

    test "a monosemous word resolves with full confidence" do
      assert {:ok, %{confidence: 1.0, domain: :noun_animal, synset_id: 102_085_443}} =
               Lexicon.disambiguate("aardvark", nil, ~w(burrow ant))
    end

    test "an unknown word cannot be disambiguated" do
      assert {:oov, nil, 0.0} = Lexicon.disambiguate("zorbl", nil, ~w(some context))
    end
  end
end
