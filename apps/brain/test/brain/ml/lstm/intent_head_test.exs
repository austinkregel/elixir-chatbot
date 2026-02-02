defmodule Brain.ML.LSTM.IntentHeadTest do
  use ExUnit.Case, async: true
  
  alias Brain.ML.LSTM.IntentHead
  
  @moduletag :lstm
  
  describe "build_model/3" do
    test "builds intent head with correct dimensions" do
      input_size = 64
      num_intents = 10
      
      model = IntentHead.build_model(input_size, num_intents)
      
      # Model should be an Axon model
      assert %Axon{} = model
    end
    
    test "accepts dropout option" do
      model = IntentHead.build_model(64, 10, dropout: 0.3)
      assert %Axon{} = model
    end
  end
  
  describe "init_params/2" do
    test "initializes parameters" do
      input_size = 64
      num_intents = 10
      
      model = IntentHead.build_model(input_size, num_intents)
      params = IntentHead.init_params(model, input_size)
      
      # Params should be an Axon.ModelState or map
      assert params != nil
      param_data = if is_struct(params, Axon.ModelState), do: params.data, else: params
      assert is_map(param_data)
      assert map_size(param_data) > 0
    end
  end
  
  describe "forward/3" do
    test "produces output probabilities with correct shape" do
      input_size = 64
      num_intents = 10
      batch_size = 4
      
      model = IntentHead.build_model(input_size, num_intents)
      params = IntentHead.init_params(model, input_size)
      
      # Create sample input using Nx.Random
      key = Nx.Random.key(42)
      {sentence_vector, _key} = Nx.Random.uniform(key, shape: {batch_size, input_size}, type: :f32)
      
      output = IntentHead.forward(model, sentence_vector, params)
      
      # Output should be [batch, num_intents]
      assert Nx.shape(output) == {batch_size, num_intents}
      
      # Output should be probabilities (sum to ~1 for each sample)
      sums = Nx.sum(output, axes: [1])
      for i <- 0..(batch_size - 1) do
        sum = Nx.to_number(sums[i])
        assert_in_delta sum, 1.0, 0.01
      end
    end
  end
  
  describe "classify/4" do
    test "returns predicted intent with confidence" do
      input_size = 64
      num_intents = 3
      
      model = IntentHead.build_model(input_size, num_intents)
      params = IntentHead.init_params(model, input_size)
      
      intent_labels = ["greeting", "weather.query", "music.play"]
      key = Nx.Random.key(42)
      {sentence_vector, _key} = Nx.Random.uniform(key, shape: {input_size}, type: :f32)
      
      {intent, confidence, all_scores} = IntentHead.classify(model, sentence_vector, params, intent_labels)
      
      # Intent should be one of the labels
      assert intent in intent_labels
      
      # Confidence should be between 0 and 1
      assert confidence >= 0.0 and confidence <= 1.0
      
      # All scores should be a map with all intents
      assert map_size(all_scores) == num_intents
      assert Map.has_key?(all_scores, "greeting")
      assert Map.has_key?(all_scores, "weather.query")
      assert Map.has_key?(all_scores, "music.play")
    end
  end
  
  describe "compute_loss/2" do
    test "computes cross-entropy loss" do
      # batch_size = 4, num_intents = 3
      
      # Create sample predictions and targets
      predictions = Nx.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.33, 0.33, 0.34]
      ], type: :f32)
      
      targets = Nx.tensor([0, 1, 2, 0], type: :s64)
      
      loss = IntentHead.compute_loss(predictions, targets)
      
      # Loss should be a scalar
      assert Nx.shape(loss) == {}
      
      # Loss should be positive
      assert Nx.to_number(loss) > 0
    end
    
    test "returns lower loss for better predictions" do
      # Good predictions
      good_preds = Nx.tensor([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05]
      ], type: :f32)
      
      # Bad predictions
      bad_preds = Nx.tensor([
        [0.1, 0.45, 0.45],
        [0.45, 0.1, 0.45]
      ], type: :f32)
      
      targets = Nx.tensor([0, 1], type: :s64)
      
      good_loss = IntentHead.compute_loss(good_preds, targets) |> Nx.to_number()
      bad_loss = IntentHead.compute_loss(bad_preds, targets) |> Nx.to_number()
      
      assert good_loss < bad_loss
    end
  end
  
  describe "top_k/5" do
    test "returns top k predictions" do
      input_size = 64
      num_intents = 5
      
      model = IntentHead.build_model(input_size, num_intents)
      params = IntentHead.init_params(model, input_size)
      
      intent_labels = ["a", "b", "c", "d", "e"]
      key = Nx.Random.key(42)
      {sentence_vector, _key} = Nx.Random.uniform(key, shape: {input_size}, type: :f32)
      
      top3 = IntentHead.top_k(model, sentence_vector, params, intent_labels, 3)
      
      assert length(top3) == 3
      
      # Should be sorted by score descending
      scores = Enum.map(top3, fn {_label, score} -> score end)
      assert scores == Enum.sort(scores, :desc)
    end
  end
end
