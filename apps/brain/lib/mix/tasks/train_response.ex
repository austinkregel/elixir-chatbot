defmodule Mix.Tasks.TrainResponse do
  @moduledoc """
  Train the LSTM response scoring model.
  
  ## Usage
  
      mix train_response [options]
  
  ## Options
  
    --epochs N       Number of training epochs (default: 15)
    --batch-size N   Batch size (default: 32)
    --hidden-size N  LSTM hidden dimension (default: 128)
    --lr FLOAT       Learning rate (default: 0.001)
  
  ## What This Trains
  
  The response scorer learns to evaluate query-response pairs:
  - Positive examples: actual responses from training data
  - Negative examples: mismatched query-response pairs
  
  After training, the scorer is used to:
  - Pick the best response from multiple candidates
  - Detect low-quality or irrelevant responses
  - Improve response selection accuracy
  
  ## Examples
  
      # Train with defaults
      mix train_response
  
      # Train with more epochs
      mix train_response --epochs 25
  """
  
  use Mix.Task
  require Logger
  
  @shortdoc "Train LSTM response quality scorer"
  
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        epochs: :integer,
        batch_size: :integer,
        hidden_size: :integer,
        lr: :float
      ]
    )
    
    # Skip async ML init during training to avoid conflicts
    Application.put_env(:brain, :skip_ml_init, true)
    
    # Start the application
    Mix.Task.run("app.start")
    
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("LSTM Response Scorer Training")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    Mix.shell().info("This model learns to score query-response pairs")
    Mix.shell().info("enabling better response selection.")
    Mix.shell().info("")
    
    # Build config
    config = []
    config = if opts[:epochs], do: [{:epochs, opts[:epochs]} | config], else: [{:epochs, 15} | config]
    config = if opts[:batch_size], do: [{:batch_size, opts[:batch_size]} | config], else: config
    config = if opts[:hidden_size], do: [{:hidden_size, opts[:hidden_size]} | config], else: [{:hidden_size, 128} | config]
    config = if opts[:lr], do: [{:learning_rate, opts[:lr]} | config], else: config
    
    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{Keyword.get(config, :epochs)}")
    Mix.shell().info("  Batch size: #{Keyword.get(config, :batch_size, 32)}")
    Mix.shell().info("  Hidden size: #{Keyword.get(config, :hidden_size)}")
    Mix.shell().info("  Learning rate: #{Keyword.get(config, :learning_rate, 0.001)}")
    Mix.shell().info("")
    
    start_time = System.monotonic_time(:second)
    
    case Brain.Response.LSTMResponse.train(config) do
      {:ok, _result} ->
        duration = System.monotonic_time(:second) - start_time
        
        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("")
        Mix.shell().info("  Total time: #{duration} seconds")
        Mix.shell().info("")
        Mix.shell().info("Model saved to priv/ml_models/lstm/response_scorer.term")
        Mix.shell().info("")
        Mix.shell().info("Usage:")
        Mix.shell().info("  # Score a response")
        Mix.shell().info("  LSTMResponse.score_response(\"What's the weather?\", \"It's sunny.\")")
        Mix.shell().info("")
        Mix.shell().info("  # Generate best response")
        Mix.shell().info("  LSTMResponse.generate(query, intent, entities)")
        Mix.shell().info("")
        
      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end
end
