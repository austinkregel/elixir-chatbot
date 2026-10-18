defmodule Brain.WorldsTest do
  @moduledoc """
  The worlds brain runs (`Brain.Worlds`): a world's tree, the copies it is
  seeded with and what is dropped when it stops; where worlds' directories
  are; and the reports a world's manager records.
  """

  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Worlds
  alias Brain.Worlds.{Embedder, EntityCandidate}

  describe "base_path/0 and world_path/1" do
    # With `:test_world_sandbox` off, `base_path/0` is `:training_worlds_path`,
    # which points at a directory of this test's own for the duration.
    setup do
      dir = Path.join(System.tmp_dir!(), "worlds_test_#{System.unique_integer([:positive])}")
      original_sandbox = Application.fetch_env(:brain, :test_world_sandbox)
      original_path = Application.fetch_env(:brain, :training_worlds_path)

      Application.put_env(:brain, :test_world_sandbox, false)
      Application.put_env(:brain, :training_worlds_path, dir)

      on_exit(fn ->
        restore_env(:test_world_sandbox, original_sandbox)
        restore_env(:training_worlds_path, original_path)
      end)

      {:ok, dir: dir}
    end

    test "base_path/0 returns temp dir when test_world_sandbox enabled" do
      Application.put_env(:brain, :test_world_sandbox, true)

      path = Worlds.base_path()
      assert String.contains?(path, "chat_bot_test_worlds")
    end

    test "base_path/0 returns training_worlds_path when test_world_sandbox is off", %{dir: dir} do
      assert Worlds.base_path() == dir
    end

    test "base_path/0 raises when no directory is configured" do
      Application.delete_env(:brain, :training_worlds_path)

      assert_raise RuntimeError, ~r/config :brain, :training_worlds_path must name the directory/, fn ->
        Worlds.base_path()
      end
    end

    test "world_path/1 joins base path with world id" do
      path = Worlds.world_path("test_my_world")
      assert String.ends_with?(path, "test_my_world")
    end
  end

  describe "a world's tree" do
    setup tags do
      # Starting a world's tree creates its graphs in Atlas, in this process.
      owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
      on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

      setup_world_sandbox()

      world_id = "test_worlds_tree_#{System.unique_integer([:positive])}"
      on_exit(fn -> Worlds.stop!(world_id) end)

      {:ok, world_id: world_id}
    end

    test "an ephemeral world created runs with its graphs and its seeded copies held in memory",
         %{world_id: world_id} do
      assert Worlds.fetch(world_id) == {:error, :not_found}

      assert :ok = Worlds.create!(world_id, :ephemeral)

      assert Worlds.fetch(world_id) == {:ok, :ephemeral}
      assert Worlds.running?(world_id)
      assert world_id in Worlds.list()
      assert Enum.all?(Atlas.Graph.world_graphs(world_id), &Atlas.Graph.graph_exists?/1)
      assert Embedder.ready?(world_id)
      refute File.exists?(Worlds.world_path(world_id))
    end

    test "stopping it drops what is loaded for it, and stopping it again is :ok", %{world_id: world_id} do
      :ok = Worlds.create!(world_id, :ephemeral)

      assert :ok = Worlds.stop!(world_id)

      assert Worlds.fetch(world_id) == {:error, :not_found}
      refute Embedder.ready?(world_id)
      assert :ok = Worlds.stop!(world_id)
    end

    test "starting a running world again is :ok in its mode and raises in the other", %{world_id: world_id} do
      :ok = Worlds.create!(world_id, :ephemeral)

      assert :ok = Worlds.start!(world_id, :ephemeral)

      assert_raise ArgumentError, ~r/runs as :ephemeral, not :persistent/, fn ->
        Worlds.start!(world_id, :persistent)
      end
    end

    test "a mode that is not a world's raises", %{world_id: world_id} do
      assert_raise ArgumentError, ~r/a world's mode is one of/, fn -> Worlds.start!(world_id, :temporary) end
    end
  end

  describe "report_candidate!/2" do
    test "broadcasts the candidate, validated in the caller" do
      :ok = Worlds.subscribe_reports()

      assert :ok =
               Worlds.report_candidate!("test_worlds_reports", %{
                 value: "Zed",
                 inferred_type: "person",
                 confidence: 0.7,
                 context: "Zed said hello"
               })

      assert_receive {:world_entity_candidate, "test_worlds_reports", %EntityCandidate{value: "Zed", context: "Zed said hello"}}
    end

    test "raises for a candidate without the text it was seen in, and reports nothing" do
      :ok = Worlds.subscribe_reports()

      assert_raise ArgumentError, ~r/missing \[:context\]/, fn ->
        Worlds.report_candidate!("test_worlds_reports", %{value: "Zed", inferred_type: "person", confidence: 0.7})
      end

      refute_received {:world_entity_candidate, _, _}
    end
  end

  defp restore_env(key, {:ok, value}), do: Application.put_env(:brain, key, value)
  defp restore_env(key, :error), do: Application.delete_env(:brain, key)
end
