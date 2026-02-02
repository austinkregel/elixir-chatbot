defmodule Brain.ML.LSTM.AccuracyComparisonTest do
  @moduledoc """
  Compares LSTM intent classifier accuracy against TF-IDF baseline.
  
  This test is marked as :slow because it trains models from scratch.
  Run with: mix test --only slow
  """
  use ExUnit.Case, async: false
  
  alias Brain.ML.LSTM.Trainer, as: LSTMTrainer
  alias Brain.ML.SimpleClassifier
  alias Brain.ML.DataLoaders
  
  @moduletag :lstm
  @moduletag :slow
  @moduletag timeout: 300_000
  
  # Test cases that historically caused misclassification
  @confusable_cases [
    # These should be weather.query, not meta.self_knowledge
    {"tell me about the weather", "weather"},
    {"can you please tell me about the weather", "weather"},
    {"what do you know about the weather", "weather"},
    {"tell me what the weather is like", "weather"},
    
    # These should be meta.self_knowledge, not weather
    {"tell me about yourself", "meta.self_knowledge"},
    {"what can you tell me about yourself", "meta.self_knowledge"},
    {"tell me about you", "meta.self_knowledge"},
    
    # Music queries
    {"play some music", "music"},
    {"can you play me some songs", "music"},
    
    # Device control
    {"turn on the lights", "device"},
    {"turn off the living room light", "device"}
  ]
  
  describe "LSTM vs TF-IDF comparison" do
    @tag :comparison
    test "LSTM handles confusable cases better than TF-IDF" do
      # Load training data
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm()
      
      # Train TF-IDF classifier
      tfidf_training_data = 
        examples
        |> Enum.map(fn ex -> {Enum.join(ex.tokens, " "), ex.intent} end)
      
      tfidf_model = SimpleClassifier.train(tfidf_training_data)
      
      # Train LSTM classifier (minimal epochs for testing)
      {:ok, lstm_model} = LSTMTrainer.train_intent_classifier(
        epochs: 3,
        batch_size: 32,
        max_seq_length: 30
      )
      
      # Test confusable cases
      tfidf_correct = 0
      lstm_correct = 0
      
      results = 
        Enum.map(@confusable_cases, fn {text, expected_domain} ->
          # TF-IDF classification
          {:ok, tfidf_intent, tfidf_conf, _} = SimpleClassifier.classify_with_details(text, tfidf_model)
          tfidf_matches = intent_matches_domain?(tfidf_intent, expected_domain)
          
          # LSTM classification
          {lstm_intent, lstm_conf, _} = LSTMTrainer.classify(text, lstm_model)
          lstm_matches = intent_matches_domain?(lstm_intent, expected_domain)
          
          %{
            text: text,
            expected_domain: expected_domain,
            tfidf_intent: tfidf_intent,
            tfidf_conf: tfidf_conf,
            tfidf_correct: tfidf_matches,
            lstm_intent: lstm_intent,
            lstm_conf: lstm_conf,
            lstm_correct: lstm_matches
          }
        end)
      
      tfidf_correct = Enum.count(results, & &1.tfidf_correct)
      lstm_correct = Enum.count(results, & &1.lstm_correct)
      
      # Log results for analysis
      IO.puts("\n=== Intent Classification Comparison ===")
      IO.puts("TF-IDF correct: #{tfidf_correct}/#{length(@confusable_cases)}")
      IO.puts("LSTM correct:   #{lstm_correct}/#{length(@confusable_cases)}")
      IO.puts("")
      
      Enum.each(results, fn r ->
        tfidf_mark = if r.tfidf_correct, do: "✓", else: "✗"
        lstm_mark = if r.lstm_correct, do: "✓", else: "✗"
        
        IO.puts("\"#{r.text}\"")
        IO.puts("  Expected: #{r.expected_domain}")
        IO.puts("  TF-IDF:   #{r.tfidf_intent} (#{Float.round(r.tfidf_conf, 3)}) #{tfidf_mark}")
        IO.puts("  LSTM:     #{r.lstm_intent} (#{Float.round(r.lstm_conf, 3)}) #{lstm_mark}")
        IO.puts("")
      end)
      
      # We expect LSTM to do at least as well as TF-IDF
      # In practice, LSTM should do better on confusable cases
      assert lstm_correct >= tfidf_correct, 
        "LSTM (#{lstm_correct}) should match or exceed TF-IDF (#{tfidf_correct}) on confusable cases"
    end
  end
  
  # Helper to check if intent matches expected domain
  defp intent_matches_domain?(intent, domain) do
    intent_lower = String.downcase(intent)
    domain_lower = String.downcase(domain)
    
    cond do
      String.contains?(intent_lower, domain_lower) -> true
      domain == "weather" and String.contains?(intent_lower, "weather") -> true
      domain == "music" and (String.contains?(intent_lower, "music") or String.contains?(intent_lower, "play")) -> true
      domain == "device" and (String.contains?(intent_lower, "device") or String.contains?(intent_lower, "light") or String.contains?(intent_lower, "heating")) -> true
      domain == "meta.self_knowledge" and String.contains?(intent_lower, "meta") -> true
      true -> false
    end
  end
end
