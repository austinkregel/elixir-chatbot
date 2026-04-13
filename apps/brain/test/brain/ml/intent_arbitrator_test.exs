defmodule Brain.ML.IntentArbitratorTest do
  @moduledoc """
  Tests for the IntentArbitrator stacked meta-learner.
  Validates feature extraction, model training, arbitration decisions,
  and model save/load round-trips.
  """
  use ExUnit.Case, async: false

  alias Brain.ML.IntentArbitrator

  describe "extract_features/1" do
    test "returns 32-element feature vector with known inputs" do
      features =
        IntentArbitrator.extract_features(%{
          lstm: %{intent: "greeting.hello", confidence: 0.9, scores: [{"greeting.hello", 0.9}, {"farewell.bye", 0.05}]},
          tfidf: %{intent: "greeting.hello", confidence: 0.85, scores: [{"greeting.hello", 0.85}, {"farewell.bye", 0.1}]},
          text: "hello there"
        })

      assert length(features) == 32
      assert Enum.all?(features, &is_number/1)

      [lstm_conf, tfidf_conf, lstm_margin, tfidf_margin, agree | _rest] = features
      assert_in_delta lstm_conf, 0.9, 0.01
      assert_in_delta tfidf_conf, 0.85, 0.01
      assert_in_delta lstm_margin, 0.85, 0.01
      assert_in_delta tfidf_margin, 0.75, 0.01
      assert agree == 1.0
    end

    test "returns 32-element vector with disagreement" do
      features =
        IntentArbitrator.extract_features(%{
          lstm: %{intent: "weather.query", confidence: 0.7, scores: []},
          tfidf: %{intent: "greeting.hello", confidence: 0.6, scores: []},
          text: "what's the weather"
        })

      assert length(features) == 32
      [_lstm_conf, _tfidf_conf, _lstm_margin, _tfidf_margin, agree | _] = features
      assert agree == 0.0
    end

    test "handles nil/missing subsystem signals gracefully" do
      features =
        IntentArbitrator.extract_features(%{
          lstm: %{intent: "test", confidence: 0.5, scores: []},
          tfidf: nil,
          text: nil,
          structural: nil,
          keyword: nil,
          memory: nil,
          sentiment: nil,
          discourse: nil
        })

      assert length(features) == 32
      assert Enum.all?(features, &is_number/1)
    end

    test "handles completely empty input" do
      features = IntentArbitrator.extract_features(%{})
      assert length(features) == 32
      assert Enum.all?(features, &is_number/1)
    end

    test "structural signals are correctly encoded" do
      features =
        IntentArbitrator.extract_features(%{
          lstm: %{intent: "test", confidence: 0.5, scores: []},
          structural: %{is_question: true, is_imperative: false, has_modal: true, is_declarative: false}
        })

      assert Enum.at(features, 8) == 1.0
      assert Enum.at(features, 9) == 0.0
      assert Enum.at(features, 10) == 1.0
      assert Enum.at(features, 11) == 0.0
    end

    test "short utterance flag is set for 3 words or fewer" do
      short = IntentArbitrator.extract_features(%{text: "hi there"})
      long = IntentArbitrator.extract_features(%{text: "what is the weather in new york today"})

      assert Enum.at(short, 13) == 1.0
      assert Enum.at(long, 13) == 0.0
    end

    test "domain depth is calculated correctly" do
      features =
        IntentArbitrator.extract_features(%{
          lstm: %{intent: "music.player.skip_forward", confidence: 0.8, scores: []},
          tfidf: %{intent: "music", confidence: 0.5, scores: []}
        })

      lstm_depth = Enum.at(features, 20)
      tfidf_depth = Enum.at(features, 21)
      same_domain = Enum.at(features, 22)

      assert_in_delta lstm_depth, 0.5, 0.01
      assert_in_delta tfidf_depth, 0.0, 0.01
      assert same_domain == 1.0
    end
  end

  describe "build_model/0" do
    test "builds a valid Axon model" do
      model = IntentArbitrator.build_model()
      assert %Axon{} = model
    end
  end

  describe "train/2" do
    test "trains a model on synthetic data" do
      training_data = generate_synthetic_training_data(50)

      case IntentArbitrator.train(training_data, epochs: 3, batch_size: 8) do
        {:ok, model, params} ->
          assert %Axon{} = model
          assert params != nil

        {:error, :insufficient_training_data} ->
          flunk("Should have enough data with 50 examples")
      end
    end

    test "returns error with insufficient data" do
      result = IntentArbitrator.train([], epochs: 1, batch_size: 8)
      assert {:error, :insufficient_training_data} = result
    end
  end

  describe "model save/load round-trip" do
    test "saved model produces same predictions after reload" do
      training_data = generate_synthetic_training_data(50)

      {:ok, _model, params} = IntentArbitrator.train(training_data, epochs: 3, batch_size: 8)

      tmp_dir = System.tmp_dir!()
      test_path = Path.join(tmp_dir, "test_arbitrator_#{System.unique_integer([:positive])}.term")

      original_path_fn = fn -> test_path end

      portable =
        if is_struct(params, Axon.ModelState) do
          Axon.ModelState.new(transfer_params(params.data))
        else
          transfer_params(params)
        end

      data = %{params: portable, feature_count: 32}
      File.write!(test_path, :erlang.term_to_binary(data))

      loaded_data = :erlang.binary_to_term(File.read!(test_path))
      loaded_model = IntentArbitrator.build_model()

      test_features = Enum.at(training_data, 0).features
      input = Nx.tensor([test_features], type: :f32)

      original_pred = Axon.predict(loaded_model, params, %{"features" => input}, compiler: EXLA)
      loaded_pred = Axon.predict(loaded_model, loaded_data.params, %{"features" => input}, compiler: EXLA)

      original_list = Nx.to_flat_list(original_pred)
      loaded_list = Nx.to_flat_list(loaded_pred)

      Enum.zip(original_list, loaded_list)
      |> Enum.each(fn {orig, loaded} ->
        assert_in_delta orig, loaded, 1.0e-5
      end)

      File.rm(test_path)
    end
  end

  # --- Helpers ---

  defp generate_synthetic_training_data(count) do
    for _i <- 1..count do
      lstm_conf = :rand.uniform()
      tfidf_conf = :rand.uniform()
      agree = if :rand.uniform() > 0.5, do: 1.0, else: 0.0

      features = [
        lstm_conf,
        tfidf_conf,
        :rand.uniform() * 0.5,
        :rand.uniform() * 0.5,
        agree,
        lstm_conf - tfidf_conf,
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        0.0,
        0.0,
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        :rand.uniform(),
        if(:rand.uniform() > 0.7, do: 1.0, else: 0.0),
        min(:rand.uniform() * 20 / 50.0, 1.0),
        :rand.uniform(),
        :rand.uniform() * 0.4,
        :rand.uniform(),
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        :rand.uniform(),
        :rand.uniform() * 0.5,
        :rand.uniform() * 0.5,
        agree,
        # Entity/slot coherence features (9)
        :rand.uniform() * 0.6,
        :rand.uniform(),
        :rand.uniform(),
        :rand.uniform(),
        :rand.uniform(),
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        if(:rand.uniform() > 0.5, do: 1.0, else: 0.0),
        :rand.uniform(),
        :rand.uniform()
      ]

      label = if lstm_conf > tfidf_conf + 0.1, do: :lstm, else: :tfidf

      %{features: features, label: label}
    end
  end

  defp transfer_params(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
  defp transfer_params(%Axon.ModelState{} = s), do: Axon.ModelState.new(transfer_params(s.data))
  defp transfer_params(map) when is_map(map), do: Map.new(map, fn {k, v} -> {k, transfer_params(v)} end)
  defp transfer_params(other), do: other
end
