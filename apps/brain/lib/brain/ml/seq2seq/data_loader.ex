defmodule Brain.ML.Seq2Seq.DataLoader do
  @moduledoc """
  Loads training data from multiple sources for seq2seq models.
  
  Combines:
  1. Intent data (data/intents/*.json) - training phrases as source, intent descriptions as target
  2. Task benchmark data (data/domain_specific_tasks/*.json) - questions/passages as source, answers as target
  3. World conversation data (priv/training_worlds/*/events.jsonl) - conversations as source, summaries as target
  """
  
  require Logger
  alias Brain.ML.Tokenizer
  
  @type training_pair :: {String.t(), String.t()}  # {source, target}
  
  # ============================================================================
  # Public API
  # ============================================================================
  
  @doc """
  Load all training data for a given world.
  
  Returns a list of {source, target} pairs.
  """
  def load_training_data(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, "default")
    
    intent_pairs = load_intent_pairs()
    task_pairs = load_task_pairs()
    world_pairs = load_world_conversation_pairs(world_id)
    
    all_pairs = intent_pairs ++ task_pairs ++ world_pairs
    
    Logger.info("Loaded training data", %{
      intent_pairs: length(intent_pairs),
      task_pairs: length(task_pairs),
      world_pairs: length(world_pairs),
      total: length(all_pairs)
    })
    
    all_pairs
  end
  
  @doc """
  Load training data for a specific world.
  """
  def load_for_world(world_id) when is_binary(world_id) do
    load_training_data(world_id: world_id)
  end
  
  # ============================================================================
  # Intent Data Loading
  # ============================================================================
  
  defp load_intent_pairs do
    intents_path = Application.get_env(:brain, :ml)[:training_data_path]
    intents_dir = Path.join(intents_path, "intents")
    
    case File.ls(intents_dir) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.ends_with?(&1, ".json"))
        |> Enum.filter(&(!String.contains?(&1, "_usersays_")))  # Skip user says files
        |> Enum.flat_map(&load_intent_file(Path.join(intents_dir, &1)))
      
      {:error, _} ->
        Logger.warning("Intent directory not found: #{intents_dir}")
        []
    end
  end
  
  defp load_intent_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, intent_data} ->
            _intent_name = Map.get(intent_data, "name", "")
            intent_desc = build_intent_description(intent_data)
            
            # Load corresponding user says file
            user_says_path = String.replace(path, ".json", "_usersays_en.json")
            user_says = load_user_says(user_says_path)
            
            # Create pairs: user says -> intent description
            Enum.map(user_says, fn user_say ->
              {user_say, intent_desc}
            end)
          
          {:error, _} ->
            []
        end
      
      {:error, _} ->
        []
    end
  end
  
  defp load_user_says(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            Enum.map(data, fn item ->
              Map.get(item, "data", [])
              |> Enum.map(&Map.get(&1, "text", ""))
              |> Enum.join("")
            end)
          
          _ ->
            []
        end
      
      {:error, _} ->
        []
    end
  end
  
  defp build_intent_description(intent_data) do
    name = Map.get(intent_data, "name", "")
    
    # Try to extract description from responses
    responses = Map.get(intent_data, "responses", [])
    
    case responses do
      [first_response | _] ->
        messages = Map.get(first_response, "messages", [])
        
        case messages do
          [first_message | _] ->
            speech = Map.get(first_message, "speech", [])
            
            case speech do
              [first_speech | _] when is_binary(first_speech) ->
                first_speech
            
              _ ->
                "User wants to #{name}"
            end
          
          _ ->
            "User wants to #{name}"
        end
      
      _ ->
        "User wants to #{name}"
    end
  end
  
  # ============================================================================
  # Task Benchmark Data Loading
  # ============================================================================
  
  defp load_task_pairs do
    tasks_path = Application.get_env(:brain, :ml)[:training_data_path]
    tasks_dir = Path.join(tasks_path, "domain_specific_tasks")
    
    case File.ls(tasks_dir) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.ends_with?(&1, ".json"))
        |> Enum.take(100)  # Limit to first 100 tasks for performance
        |> Enum.flat_map(&load_task_file(Path.join(tasks_dir, &1)))
      
      {:error, _} ->
        Logger.warning("Task directory not found: #{tasks_dir}")
        []
    end
  end
  
  defp load_task_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, task_data} ->
            extract_task_pairs(task_data)
          
          {:error, _} ->
            []
        end
      
      {:error, _} ->
        []
    end
  end
  
  defp extract_task_pairs(task_data) do
    # Try different task formats
    cond do
      # Question answering format: input -> output
      Map.has_key?(task_data, "input") and Map.has_key?(task_data, "output") ->
        input = Map.get(task_data, "input", "")
        output = Map.get(task_data, "output", "")
        
        if input != "" and output != "" do
          [{input, output}]
        else
          []
        end
      
      # Examples format: list of examples
      Map.has_key?(task_data, "examples") ->
        examples = Map.get(task_data, "examples", [])
        
        Enum.flat_map(examples, fn example ->
          input = Map.get(example, "input", Map.get(example, "question", ""))
          output = Map.get(example, "output", Map.get(example, "target", Map.get(example, "answer", "")))
          
          if input != "" and output != "" do
            [{input, output}]
          else
            []
          end
        end)
      
      # Summarization format: passage -> summary
      Map.has_key?(task_data, "passage") and Map.has_key?(task_data, "summary") ->
        passage = Map.get(task_data, "passage", "")
        summary = Map.get(task_data, "summary", "")
        
        if passage != "" and summary != "" do
          [{passage, summary}]
        else
          []
        end
      
      true ->
        []
    end
  end
  
  # ============================================================================
  # World Conversation Data Loading
  # ============================================================================
  
  defp load_world_conversation_pairs(world_id) do
    if world_id == "default" do
      []
    else
      world_path = Path.join(["priv", "training_worlds", world_id, "events.jsonl"])
      
      if File.exists?(world_path) do
        load_events_jsonl(world_path)
      else
        []
      end
    end
  end
  
  defp load_events_jsonl(path) do
    case File.stream!(path) do
      stream ->
        stream
        |> Stream.map(&Jason.decode/1)
        |> Stream.filter(&match?({:ok, _}, &1))
        |> Stream.map(fn {:ok, event} -> event end)
        |> Stream.filter(&Map.has_key?(&1, "text"))
        |> Stream.chunk_every(5)  # Group events into conversation chunks
        |> Stream.flat_map(&build_conversation_pair/1)
        |> Enum.to_list()
    end
  rescue
    _ ->
      []
  end
  
  defp build_conversation_pair(events) when is_list(events) do
    # Combine events into a conversation
    conversation = 
      events
      |> Enum.map(&Map.get(&1, "text", ""))
      |> Enum.reject(&(&1 == ""))
      |> Enum.join(" ")
    
    # Create a simple summary (first 50 words)
    summary = 
      conversation
      |> Tokenizer.tokenize_normalized()
      |> Enum.take(50)
      |> Enum.join(" ")
    
    if conversation != "" and summary != "" do
      [{conversation, summary}]
    else
      []
    end
  end
  
  defp build_conversation_pair(_), do: []
  
  # ============================================================================
  # Data Preprocessing
  # ============================================================================
  
  @doc """
  Preprocess training pairs: normalize, filter by length, etc.
  """
  def preprocess_pairs(pairs, opts \\ []) do
    max_source_length = Keyword.get(opts, :max_source_length, 100)
    max_target_length = Keyword.get(opts, :max_target_length, 50)
    min_length = Keyword.get(opts, :min_length, 5)
    
    pairs
    |> Enum.map(fn {source, target} ->
      {normalize_text(source), normalize_text(target)}
    end)
    |> Enum.filter(fn {source, target} ->
      source_tokens = Tokenizer.tokenize_normalized(source)
      target_tokens = Tokenizer.tokenize_normalized(target)
      
      length(source_tokens) >= min_length and
      length(source_tokens) <= max_source_length and
      length(target_tokens) >= min_length and
      length(target_tokens) <= max_target_length
    end)
  end
  
  defp normalize_text(text) do
    text
    |> String.downcase()
    |> String.trim()
  end
end
