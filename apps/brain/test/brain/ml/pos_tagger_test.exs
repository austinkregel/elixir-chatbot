defmodule Brain.ML.POSTaggerTest do
  @moduledoc """
  The tagger is measured against held-out treebank sentences, not against
  its own internals. The model under test is the one `ModelFactory` trains at
  suite start on a slice of the EWT fixtures.
  """
  use ExUnit.Case, async: false

  alias Brain.ML.POSTagger
  alias Brain.Training.{Fixture, POS}

  @moduletag :tmp_dir

  setup_all do
    {:ok, model} = POSTagger.load_model()
    test = POS.fixture_paths() |> Map.fetch!("test") |> Fixture.load!() |> Fixture.pos_sequences()
    train = POS.fixture_paths() |> Map.fetch!("train") |> Fixture.load!() |> Fixture.pos_sequences()
    # Not `test:`: ExUnit puts the test's name under that key.
    {:ok, model: model, test_split: test, train: train}
  end

  describe "the trained model" do
    test "beats tagging each word with its most frequent training tag", %{model: model, test_split: test, train: train} do
      # The test split is never trained on; its last 500 sentences. The lookup
      # learns from the same sentences the model did: the first
      # `training.sequences` of the train split (Brain.Training.POS :limit).
      held_out = Enum.take(test, -500)
      seen = Enum.take(train, model.training.sequences)
      report = POSTagger.evaluate(held_out, model)

      assert report.accuracy > POS.lookup_baseline(seen, held_out)
    end

    test "records its evaluation and the data it was trained on", %{model: model} do
      e = model.evaluation

      assert e.accuracy > e.lookup_baseline
      assert e.split == "ud_ewt.test"
      assert is_map(e.confusion) and map_size(e.per_tag) > 0

      names = Enum.map(model.training.inputs, & &1.name)
      assert names == ~w(ud_ewt.train.json ud_ewt.dev.json ud_ewt.test.json)
      assert Enum.all?(model.training.inputs, &(byte_size(&1.sha256) == 64))
    end

    test "tags a word by its context, not by the word alone", %{model: model, test_split: test} do
      # Words the treebank itself tags differently in different sentences. A
      # lookup table gives each word one tag; a tagger that reads context
      # gives some of these words more than one.
      occurrences = Enum.flat_map(test, fn seq -> Enum.zip(seq.tokens, seq.tags) |> Enum.map(&{seq, &1}) end)

      ambiguous =
        occurrences
        |> Enum.group_by(fn {_seq, {token, _tag}} -> String.downcase(token) end)
        |> Enum.filter(fn {_w, occ} -> occ |> Enum.map(fn {_s, {_t, tag}} -> tag end) |> Enum.uniq() |> length() > 1 end)
        |> Enum.take(40)

      assert length(ambiguous) >= 20

      varied =
        Enum.count(ambiguous, fn {word, occ} ->
          occ
          |> Enum.map(fn {seq, _} ->
            POSTagger.predict(seq.tokens, model)
            |> Enum.find_value(fn {t, tag} -> if String.downcase(t) == word, do: tag end)
          end)
          |> Enum.uniq()
          |> length() > 1
        end)

      assert varied > 0, "every ambiguous word got one tag everywhere: the model ignores context"
    end

    test "every predicted tag is in the tagset, one per token, unknown words included", %{model: model} do
      tokens = ["Zorblax", "quibbled", "the", "flurm", "."]
      tags = POSTagger.predict_tags(tokens, model)

      assert length(tags) == length(tokens)
      assert Enum.all?(tags, &(&1 in POSTagger.valid_tags()))
      assert POSTagger.predict(tokens, model) == Enum.zip(tokens, tags)
    end

    test "tags nothing for no tokens", %{model: model} do
      assert POSTagger.predict([], model) == []
      assert POSTagger.predict_tags([], model) == []
    end
  end

  describe "train/2" do
    test "an empty training set is an error" do
      assert {:error, _} = POSTagger.train([])
    end

    test "a sequence whose tags do not match its tokens raises, not filtered out" do
      assert_raise ArgumentError, ~r/as many tags as tokens/, fn ->
        POSTagger.train([%{tokens: ["I", "am"], tags: ["PRON"]}])
      end
    end

    test "a tag outside the tagset raises" do
      assert_raise ArgumentError, ~r/outside the tagset \["CCONJ"\]/, fn ->
        POSTagger.train([%{tokens: ["and"], tags: ["CCONJ"]}])
      end
    end

    test "reports every epoch, and with patience nil never stops early", %{train: train} do
      parent = self()
      on_epoch = fn progress, model_at -> send(parent, {:epoch, progress, model_at}) end

      {:ok, model} =
        POSTagger.train(Enum.take(train, 40),
          dev: Enum.slice(train, 40, 20),
          seed: 7,
          config: [max_epochs: 3, patience: nil],
          on_epoch: on_epoch
        )

      epochs =
        for n <- 1..3 do
          assert_received {:epoch, %{epoch: ^n} = progress, model_at}
          assert is_float(progress.loss) and progress.loss > 0
          assert is_float(progress.dev_accuracy)
          {progress, model_at}
        end

      assert model.training.epochs_run == 3
      {last, model_at} = List.last(epochs)
      snapshot = model_at.()
      assert snapshot.training.epochs_run == 3
      assert length(POSTagger.predict_tags(["I", "may", "go"], snapshot)) == 3
      assert last.best_epoch == model.training.best_epoch
    end

    test "is deterministic from its seed", %{train: train} do
      small = Enum.take(train, 40)
      opts = [seed: 7, config: [max_epochs: 1]]

      {:ok, a} = POSTagger.train(small, opts)
      {:ok, b} = POSTagger.train(small, opts)

      assert a.params.data == b.params.data
      assert POSTagger.predict_tags(["I", "may", "go"], a) == POSTagger.predict_tags(["I", "may", "go"], b)
    end
  end

  describe "persistence" do
    test "saves and loads a model", %{model: model, tmp_dir: dir} do
      path = Path.join(dir, "pos_model.term")

      assert {:ok, ^path} = POSTagger.save_model(model, path)
      assert {:ok, loaded} = POSTagger.load_model(path)
      assert loaded.vocab == model.vocab
      assert POSTagger.predict_tags(["turn", "it", "off"], loaded) == POSTagger.predict_tags(["turn", "it", "off"], model)
    end

    test "refuses a model of another format", %{tmp_dir: dir} do
      path = Path.join(dir, "old.term")
      File.write!(path, :erlang.term_to_binary(%{tag_vocabulary: %{}, feature_weights: %{}}))

      assert {:error, message} = POSTagger.load_model(path)
      assert message =~ "not a POS model of format"
    end

    test "a missing file is an error" do
      assert {:error, _} = POSTagger.load_model("/nonexistent/path/model.term")
    end
  end
end
