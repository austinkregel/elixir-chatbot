defmodule Mix.Tasks.TrainUnified do
  @moduledoc """
  Train the unified LSTM model for multi-task NLP.
  
  ## Usage
  
      mix train_unified [options]
  
  ## Options
  
    --epochs N       Number of training epochs (default: 20)
    --batch-size N   Batch size (default: 32)
    --hidden-size N  LSTM hidden dimension (default: 128)
    --lr FLOAT       Learning rate (default: 0.001)
    --name NAME      Experiment name for A/B testing
  
  ## Examples
  
      # Train with defaults
      mix train_unified
  
      # Train with more epochs
      mix train_unified --epochs 30 --name "unified_30ep"
  
  This trains a shared LSTM encoder that powers:
  - Intent classification
  - Named Entity Recognition (NER)  
  - Sentiment analysis
  - Speech act classification
  """
  
  use Mix.Task
  require Logger
  
  @shortdoc "Train unified multi-task LSTM model"
  
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        epochs: :integer,
        batch_size: :integer,
        hidden_size: :integer,
        embedding_size: :integer,
        lr: :float,
        name: :string
      ]
    )
    
    # Skip async ML init during training to avoid conflicts
    Application.put_env(:brain, :skip_ml_init, true)
    
    # Start the application
    Mix.Task.run("app.start")
    
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Unified Multi-Task LSTM Training (EXLA Accelerated)")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    Mix.shell().info("This model powers:")
    Mix.shell().info("  - Intent Classification")
    Mix.shell().info("  - Named Entity Recognition")
    Mix.shell().info("  - Sentiment Analysis")
    Mix.shell().info("  - Speech Act Classification")
    Mix.shell().info("")
    
    # Build config from options
    config = []
    config = if opts[:epochs], do: [{:epochs, opts[:epochs]} | config], else: [{:epochs, 20} | config]
    config = if opts[:batch_size], do: [{:batch_size, opts[:batch_size]} | config], else: config
    config = if opts[:hidden_size], do: [{:hidden_size, opts[:hidden_size]} | config], else: [{:hidden_size, 128} | config]
    config = if opts[:embedding_size], do: [{:embedding_size, opts[:embedding_size]} | config], else: [{:embedding_size, 128} | config]
    config = if opts[:lr], do: [{:learning_rate, opts[:lr]} | config], else: config
    config = if opts[:name], do: [{:name, opts[:name]} | config], else: config
    
    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{Keyword.get(config, :epochs)}")
    Mix.shell().info("  Batch size: #{Keyword.get(config, :batch_size, 32)}")
    Mix.shell().info("  Hidden size: #{Keyword.get(config, :hidden_size)}")
    Mix.shell().info("  Embedding size: #{Keyword.get(config, :embedding_size)}")
    Mix.shell().info("  Learning rate: #{Keyword.get(config, :learning_rate, 0.001)}")
    if opts[:name], do: Mix.shell().info("  Experiment: #{opts[:name]}")
    Mix.shell().info("")
    
    start_time = System.monotonic_time(:second)
    
    case Brain.ML.LSTM.UnifiedModel.train(config) do
      {:ok, result} ->
        duration = System.monotonic_time(:second) - start_time
        
        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("")
        Mix.shell().info("  Total time: #{duration} seconds")
        Mix.shell().info("  Vocabulary size: #{map_size(result.vocabularies.token_vocab)}")
        Mix.shell().info("  Intent classes: #{map_size(result.vocabularies.intent_to_idx)}")
        Mix.shell().info("")
        Mix.shell().info("Model saved to priv/ml_models/lstm/unified_model.term")
        Mix.shell().info("")
        Mix.shell().info("Usage:")
        Mix.shell().info("  Brain.ML.LSTM.UnifiedModel.analyze(\"What's the weather?\")")
        Mix.shell().info("")
        
      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end
end
