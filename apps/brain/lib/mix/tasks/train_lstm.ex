defmodule Mix.Tasks.TrainLstm do
  @moduledoc """
  Train LSTM models using Axon's optimized training loop with EXLA.
  
  ## Usage
  
      mix train_lstm [options]
  
  ## Options
  
    --epochs N       Number of training epochs (default: 5)
    --batch-size N   Batch size (default: 32)
    --hidden-size N  LSTM hidden dimension (default: 64)
    --lr FLOAT       Learning rate (default: 0.001)
    --name NAME      Experiment name for A/B testing
    --compare        Show comparison of all experiments
  
  ## Examples
  
      # Train with defaults (fast, ~1-2 minutes)
      mix train_lstm
  
      # A/B test different configurations
      mix train_lstm --epochs 10 --name "baseline"
      mix train_lstm --epochs 10 --hidden-size 128 --name "larger_hidden"
      mix train_lstm --epochs 20 --name "more_epochs"
      
      # Compare all experiments
      mix train_lstm --compare
  
  This task uses EXLA for accelerated training. The first epoch may be slower
  due to JIT compilation, but subsequent epochs will be much faster.
  """
  
  use Mix.Task
  require Logger
  
  @shortdoc "Train LSTM intent classifier with EXLA acceleration"
  
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        epochs: :integer,
        batch_size: :integer,
        hidden_size: :integer,
        embedding_size: :integer,
        lr: :float,
        name: :string,
        compare: :boolean
      ]
    )
    
    # Skip async ML init during training to avoid conflicts
    Application.put_env(:brain, :skip_ml_init, true)
    
    # Start the application
    Mix.Task.run("app.start")
    
    # Handle --compare flag
    if opts[:compare] do
      Brain.ML.LSTM.ExperimentTracker.print_comparison()
      return_ok()
    else
      run_training(opts)
    end
  end
  
  defp run_training(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("LSTM Intent Classifier Training (EXLA Accelerated)")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    
    # Build config from options
    config = []
    config = if opts[:epochs], do: [{:epochs, opts[:epochs]} | config], else: config
    config = if opts[:batch_size], do: [{:batch_size, opts[:batch_size]} | config], else: config
    config = if opts[:hidden_size], do: [{:hidden_size, opts[:hidden_size]} | config], else: config
    config = if opts[:embedding_size], do: [{:embedding_size, opts[:embedding_size]} | config], else: config
    config = if opts[:lr], do: [{:learning_rate, opts[:lr]} | config], else: config
    config = if opts[:name], do: [{:name, opts[:name]} | config], else: config
    
    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{Keyword.get(config, :epochs, 5)}")
    Mix.shell().info("  Batch size: #{Keyword.get(config, :batch_size, 32)}")
    Mix.shell().info("  Hidden size: #{Keyword.get(config, :hidden_size, 64)}")
    Mix.shell().info("  Learning rate: #{Keyword.get(config, :learning_rate, 0.001)}")
    Mix.shell().info("  Backend: EXLA (#{get_exla_target()})")
    if opts[:name], do: Mix.shell().info("  Experiment: #{opts[:name]}")
    Mix.shell().info("")
    
    start_time = System.monotonic_time(:second)
    
    case Brain.ML.LSTM.AxonTrainer.train_intent_classifier(config) do
      {:ok, result} ->
        duration = System.monotonic_time(:second) - start_time
        metrics = result[:metrics] || %{}
        
        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("")
        Mix.shell().info("  Total time: #{duration} seconds")
        Mix.shell().info("  Epochs completed: #{metrics[:epochs_completed] || result.config.epochs}")
        Mix.shell().info("  Vocabulary size: #{map_size(result.vocabularies.token_vocab)}")
        Mix.shell().info("  Intent classes: #{map_size(result.vocabularies.intent_to_idx)}")
        Mix.shell().info("")
        
        if metrics[:best_val_accuracy] do
          Mix.shell().info("  Best validation accuracy: #{Float.round(metrics.best_val_accuracy * 100, 1)}%")
          Mix.shell().info("  Final train accuracy: #{Float.round(metrics.final_train_accuracy * 100, 1)}%")
          Mix.shell().info("  Final validation loss: #{Float.round(metrics.final_val_loss, 3)}")
          Mix.shell().info("")
        end
        
        Mix.shell().info("Model saved to priv/ml_models/lstm/axon_intent.term")
        
        if opts[:name] do
          Mix.shell().info("Experiment '#{opts[:name]}' recorded. Run 'mix train_lstm --compare' to compare.")
        end
        
        Mix.shell().info("")
        
      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end
  
  defp return_ok, do: :ok
  
  defp get_exla_target do
    case System.get_env("XLA_TARGET") do
      nil -> "cpu"
      target -> target
    end
  end
end
