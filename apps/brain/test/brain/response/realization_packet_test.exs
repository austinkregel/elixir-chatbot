defmodule Brain.Response.RealizationPacketTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.ChunkAnalysis
  alias Brain.Memory.RecurringPattern
  alias Brain.Memory.Types.{Episode, SemanticFact}
  alias Brain.Response.RealizationPacket

  describe "build/3 with the turn's memory" do
    setup do
      test_pid = self()
      handler = "realization-packet-test-#{System.unique_integer([:positive])}"

      :ok =
        :telemetry.attach(
          handler,
          [:chat_bot, :memory, :semantic_fact, :used],
          fn _event, _measurements, metadata, _config -> send(test_pid, {:used, metadata}) end,
          nil
        )

      on_exit(fn -> :telemetry.detach(handler) end)

      pattern = %RecurringPattern{
        world_id: "test_realization_packet",
        fact: SemanticFact.new("is it sunny in paris | is it sunny in london", [], ["ep1", "ep2", "ep3"], []),
        similarity: 0.8,
        reply: "It's usually sunny there.",
        reply_count: 2,
        evidence_count: 3
      }

      episode = Episode.new("is it sunny in lyon", "conversation", "Bring a coat.", [], [])

      {:ok, pattern: pattern, episode: {episode, 0.9}}
    end

    defp user_message(unified_context) do
      [_system, %{role: "user", content: content}] =
        RealizationPacket.build([], %ChunkAnalysis{text: "is it sunny in paris"}, unified_context: unified_context)

      content
    end

    test "names the recurring pattern ahead of the similar past interactions, and reports it",
         %{pattern: pattern, episode: episode} do
      content = user_message(%{memory: %{recurring_pattern: pattern, similar_episodes: [episode]}})

      assert content =~
               "A recurring pattern from past conversations (3 similar messages, similarity 0.8): " <>
                 "the user says things like \"is it sunny in paris | is it sunny in london\", " <>
                 "and the reply 2 of them got was \"It's usually sunny there.\"."

      {pattern_at, _} = :binary.match(content, "A recurring pattern")
      {episodes_at, _} = :binary.match(content, "Similar past interactions")
      assert pattern_at < episodes_at

      fact_id = pattern.fact.id
      assert_receive {:used, %{semantic_fact_id: ^fact_id, used_by: :realization_packet}}
    end

    test "names no pattern for a turn that has none", %{episode: episode} do
      content = user_message(%{memory: %{recurring_pattern: nil, similar_episodes: [episode]}})

      refute content =~ "recurring pattern"
      assert content =~ "Similar past interactions"
      refute_receive {:used, _}, 50
    end
  end
end
