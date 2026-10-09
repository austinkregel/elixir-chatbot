defmodule Brain.Response.SurfaceRealizerSynthesizerOptsTest do
  use ExUnit.Case, async: true

  alias Brain.Response.SurfaceRealizer
  alias Brain.Memory.RecurringPattern
  alias Brain.Memory.Types.{Episode, SemanticFact}

  describe "synthesizer_opts/2" do
    test "carries the turn's similar episodes and recurring pattern from the unified context's memory" do
      episode = %Episode{id: "ep-1", state: "what's the weather", outcome: "It's sunny."}

      pattern = %RecurringPattern{
        world_id: "test_surface_realizer",
        fact: %SemanticFact{id: "fact-1", evidence_ids: ["ep-1", "ep-2"]},
        similarity: 0.82,
        reply: "It's sunny.",
        reply_count: 2,
        evidence_count: 2
      }

      unified_context = %{
        memory: %{
          similar_episodes: [{episode, 0.9}],
          recurring_pattern: pattern
        }
      }

      opts = SurfaceRealizer.synthesizer_opts(%{confidence: 0.7}, unified_context: unified_context)

      assert Keyword.fetch!(opts, :similar_episodes) == [{episode, 0.9}]
      assert Keyword.fetch!(opts, :recurring_pattern) == pattern
      assert Keyword.fetch!(opts, :context) == unified_context
      assert Keyword.fetch!(opts, :confidence) == 0.7
    end

    test "a context built without memory gives no episodes and no pattern" do
      opts = SurfaceRealizer.synthesizer_opts(nil, unified_context: %{})

      assert Keyword.fetch!(opts, :similar_episodes) == []
      assert Keyword.fetch!(opts, :recurring_pattern) == nil
      assert Keyword.fetch!(opts, :confidence) == 0.5
    end
  end
end
