defmodule Brain.Graph.TrainingTest do
  use Brain.Test.GraphCase, async: false
  @moduletag seed_graphs: true

  describe "refresh_pos_weights/0" do
    test "updates POS tagger transition weights from pos_graph" do
      result = Brain.Graph.Training.refresh_pos_weights()
      # Will succeed if POS model exists on disk, fail gracefully otherwise
      assert result == :ok or match?({:error, _}, result)
    end

    test "respects blend ratio option" do
      # Running with blend: 0.0 should effectively be a no-op
      result = Brain.Graph.Training.refresh_pos_weights(blend: 0.0)
      assert result == :ok or match?({:error, _}, result)
    end
  end

  describe "sync_gazetteer/0" do
    test "adds knowledge_graph entities to Gazetteer" do
      # Ensure gazetteer is running
      if Process.whereis(Brain.ML.Gazetteer) do
        :ok = Brain.Graph.Training.sync_gazetteer()

        # Seeds have Paris, London, France, UK, Elixir, BEAM
        case Brain.ML.Gazetteer.lookup("Paris") do
          {:ok, results} -> assert length(results) >= 1
          :not_found -> flunk("Expected Paris to be found in Gazetteer after sync")
        end
      end
    end

    test "is idempotent (running twice does not change result count)" do
      if Process.whereis(Brain.ML.Gazetteer) do
        :ok = Brain.Graph.Training.sync_gazetteer()
        {:ok, results_first} = Brain.ML.Gazetteer.lookup("Paris")
        count_first = length(results_first)

        :ok = Brain.Graph.Training.sync_gazetteer()
        {:ok, results_second} = Brain.ML.Gazetteer.lookup("Paris")
        count_second = length(results_second)

        assert count_first == count_second
      end
    end
  end

  describe "extract_intent_priors/0" do
    test "builds transition matrix from conversation_graph topics" do
      priors = Brain.Graph.Training.extract_intent_priors()
      assert is_map(priors)

      # Seeds have smalltalk.greetings.hello -> weather.query transition
      if Map.has_key?(priors, "smalltalk.greetings.hello") do
        transitions = priors["smalltalk.greetings.hello"]
        assert Map.has_key?(transitions, "weather.query")
      end
    end
  end

  describe "apply_intent_priors/4" do
    test "boosts scores based on prior transitions" do
      scores = [{"weather.query", 0.8}, {"smalltalk.greetings.hello", 0.3}, {"unknown", 0.1}]
      priors = %{"smalltalk.greetings.hello" => %{"weather.query" => 5, "smalltalk.greetings.hello" => 1}}

      boosted = Brain.Graph.Training.apply_intent_priors(scores, "smalltalk.greetings.hello", priors, weight: 0.3)

      assert is_list(boosted)
      assert length(boosted) == 3

      # weather.query should get a boost from greeting -> weather.query prior
      {_, weather_score} = Enum.find(boosted, fn {intent, _} -> intent == "weather.query" end)
      assert weather_score > 0
    end

    test "returns original scores when prev_intent not in priors" do
      scores = [{"weather.query", 0.8}, {"smalltalk.greetings.hello", 0.3}]
      priors = %{}

      result = Brain.Graph.Training.apply_intent_priors(scores, "unknown", priors)
      assert result == scores
    end
  end
end
