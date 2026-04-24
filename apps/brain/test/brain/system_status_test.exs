defmodule Brain.SystemStatusTest do
  use ExUnit.Case, async: false

  alias Brain.SystemStatus

  describe "get_all/0" do
    test "returns a map with all expected keys" do
      result = SystemStatus.get_all()

      assert is_map(result)
      assert Map.has_key?(result, :embedder)
      assert Map.has_key?(result, :memory_store)
      assert Map.has_key?(result, :brain)
      assert Map.has_key?(result, :nlp_pipeline)
      assert Map.has_key?(result, :micro_classifiers)
    end
  end

  describe "get_embedder_status/0" do
    test "returns embedder status map" do
      result = SystemStatus.get_embedder_status()

      assert is_map(result)
      assert Map.has_key?(result, :running)
      assert Map.has_key?(result, :ready)
      assert Map.has_key?(result, :status)
      assert Map.has_key?(result, :label)
      assert Map.has_key?(result, :phase)
      assert Map.has_key?(result, :vocabulary_size)
      assert is_binary(result.label)
    end
  end

  describe "get_memory_store_status/0" do
    test "returns memory store status map" do
      result = SystemStatus.get_memory_store_status()

      assert is_map(result)
      assert Map.has_key?(result, :running)
      assert Map.has_key?(result, :ready)
      assert Map.has_key?(result, :episodes)
      assert Map.has_key?(result, :semantics)
      assert is_integer(result.episodes)
      assert is_integer(result.semantics)
    end
  end

  describe "get_brain_status/0" do
    test "returns brain status map" do
      result = SystemStatus.get_brain_status()

      assert is_map(result)
      assert Map.has_key?(result, :running)
      assert Map.has_key?(result, :ready)
      assert Map.has_key?(result, :status)
      assert Map.has_key?(result, :label)
    end
  end

  describe "get_nlp_pipeline_status/0" do
    test "returns NLP pipeline status with components" do
      result = SystemStatus.get_nlp_pipeline_status()

      assert is_map(result)
      assert Map.has_key?(result, :running)
      assert Map.has_key?(result, :ready)
      assert Map.has_key?(result, :components)
      assert is_map(result.components)
      assert Map.has_key?(result.components, :intent_classifier)
      assert Map.has_key?(result.components, :gazetteer)
    end
  end

  describe "get_all_genservers_status/0" do
    test "returns categorized genserver status" do
      result = SystemStatus.get_all_genservers_status()

      assert is_map(result)
      assert Map.has_key?(result, :categories)
      assert Map.has_key?(result, :subprocess_supervisor)
      assert Map.has_key?(result, :checked_at)
      assert Map.has_key?(result, :check_duration_ms)
      assert is_integer(result.check_duration_ms)
      assert %DateTime{} = result.checked_at
    end
  end

  describe "get_performance_metrics/0" do
    test "returns metrics map" do
      result = SystemStatus.get_performance_metrics()

      assert is_map(result)
    end
  end

  describe "get_health_indicators/0" do
    test "returns health indicators with score" do
      result = SystemStatus.get_health_indicators()

      assert is_map(result)
      assert Map.has_key?(result, :genservers_running)
      assert Map.has_key?(result, :genservers_total)
      assert Map.has_key?(result, :health_score)
      assert Map.has_key?(result, :health_status)
      assert Map.has_key?(result, :uptime_seconds)
      assert result.health_status in [:healthy, :degraded, :warning, :critical]
      assert is_integer(result.health_score)
      assert result.health_score >= 0 and result.health_score <= 100
    end
  end

  describe "get_utilization_report/0" do
    test "returns utilization report with categories" do
      result = SystemStatus.get_utilization_report()

      assert is_map(result)
      assert Map.has_key?(result, :idle)
      assert Map.has_key?(result, :never_used)
      assert Map.has_key?(result, :low_usage)
      assert Map.has_key?(result, :high_cost)
      assert Map.has_key?(result, :normal)
      assert Map.has_key?(result, :summary)
      assert is_map(result.summary)
      assert result.summary.total >= 0
    end
  end

  describe "all_ready?/0" do
    test "returns a boolean" do
      result = SystemStatus.all_ready?()
      assert is_boolean(result)
    end
  end

  describe "get_ml_models_status/0" do
    test "returns ML models status" do
      result = SystemStatus.get_ml_models_status()

      assert is_map(result)
      assert Map.has_key?(result, :pos_model)
      assert Map.has_key?(result, :entity_model)
      assert Map.has_key?(result, :gazetteer)
      assert Map.has_key?(result, :intent_classifier)
      assert Map.has_key?(result, :entity_extractor)
      assert Map.has_key?(result, :checked_at)
    end
  end

  describe "get_code_analysis_status/0" do
    test "returns code analysis status" do
      result = SystemStatus.get_code_analysis_status()

      assert is_map(result)
      assert Map.has_key?(result, :code_gazetteer)
      assert Map.has_key?(result, :language_grammar)
      assert Map.has_key?(result, :ready)
    end
  end

  describe "get_code_gazetteer_status/0" do
    test "returns code gazetteer status with stats" do
      result = SystemStatus.get_code_gazetteer_status()

      assert is_map(result)
      assert Map.has_key?(result, :running)
      assert Map.has_key?(result, :ready)
      assert Map.has_key?(result, :stats)
    end
  end

  describe "get_language_grammar_status/0" do
    test "returns language grammar status" do
      result = SystemStatus.get_language_grammar_status()

      assert is_map(result)
      assert Map.has_key?(result, :running)
      assert Map.has_key?(result, :ready)
      assert Map.has_key?(result, :stats)
    end
  end

  describe "get_training_worlds_status/0" do
    test "returns training worlds status" do
      result = SystemStatus.get_training_worlds_status()

      assert is_map(result)
      assert Map.has_key?(result, :manager_ready)
      assert Map.has_key?(result, :active_worlds)
      assert Map.has_key?(result, :persisted_worlds)
      assert Map.has_key?(result, :worlds)
      assert is_list(result.worlds)
    end
  end

  describe "get_readiness_details/1" do
    test "returns readiness details with defaults" do
      result = SystemStatus.get_readiness_details()

      assert is_map(result)
      assert Map.has_key?(result, :core)
      assert Map.has_key?(result, :embedder_details)
      assert Map.has_key?(result, :world_embedder)
      assert Map.has_key?(result, :world_models)
      assert Map.has_key?(result, :nlp_pipeline)
      assert Map.has_key?(result, :ml_models)
      assert Map.has_key?(result, :template_store)
      assert Map.has_key?(result, :all_ready)
    end

    test "accepts world_id option" do
      result = SystemStatus.get_readiness_details(world_id: "test_world")

      assert result.world_embedder.world_id == "test_world"
      assert result.world_models.world_id == "test_world"
    end
  end

  describe "get_world_embedder_status/1" do
    test "returns status for unknown world" do
      result = SystemStatus.get_world_embedder_status("nonexistent")

      assert result.world_id == "nonexistent"
      assert result.ready == false
      assert result.phase in [:not_initialized, :table_not_ready, :no_data]
    end
  end

  describe "get_world_models_status/1" do
    test "returns status for a world" do
      result = SystemStatus.get_world_models_status("test_world")

      assert is_map(result)
      assert result.world_id == "test_world"
      assert Map.has_key?(result, :has_models)
      assert Map.has_key?(result, :status)
    end
  end

  describe "get_micro_classifiers_status/0" do
    test "returns micro classifiers status" do
      result = SystemStatus.get_micro_classifiers_status()
      assert is_map(result)
    end
  end

  describe "get_response_timing/0" do
    test "returns response timing metrics" do
      result = SystemStatus.get_response_timing()

      assert is_map(result)
      assert Map.has_key?(result, :brain_avg_ms)
      assert Map.has_key?(result, :brain_count)
      assert Map.has_key?(result, :pipeline_avg_ms)
      assert Map.has_key?(result, :pipeline_count)
    end
  end
end
