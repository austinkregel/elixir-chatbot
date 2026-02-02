defmodule Mix.Tasks.TrainSeq2seq do
  @moduledoc """
  Train the LSTM + Attention seq2seq model.
  
  ## Examples
  
      # Train default model
      mix train_seq2seq
      
      # Train for specific world
      mix train_seq2seq --world star_trek
      
      # Train with custom epochs
      mix train_seq2seq --epochs 20
      
      # Train with custom batch size
      mix train_seq2seq --batch-size 64
  """
  
  use Mix.Task
  
  @shortdoc "Train the LSTM + Attention seq2seq model"
  
  @switches [
    world: :string,
    epochs: :integer,
    batch_size: :integer,
    learning_rate: :float
  ]
  
  @aliases [
    w: :world,
    e: :epochs,
    b: :batch_size,
    l: :learning_rate
  ]
  
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, switches: @switches, aliases: @aliases)
    
    world_id = Keyword.get(opts, :world, "default")
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    
    Mix.Task.run("app.start")
    
    require Logger
    Logger.info("Starting seq2seq training", %{
      world_id: world_id,
      epochs: epochs,
      batch_size: batch_size,
      learning_rate: learning_rate
    })
    
    case Brain.ML.Seq2Seq.Trainer.train(world_id, 
           epochs: epochs, 
           batch_size: batch_size,
           learning_rate: learning_rate
         ) do
      {:ok, _model} ->
        Logger.info("Training completed successfully")
        
        # Save the model
        case Brain.ML.Seq2Seq.save_model(world_id) do
          :ok ->
            Logger.info("Model saved successfully")
          
          {:error, reason} ->
            Logger.error("Failed to save model: #{inspect(reason)}")
        end
      
      {:error, reason} ->
        Logger.error("Training failed: #{inspect(reason)}")
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end
end
