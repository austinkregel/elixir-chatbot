defmodule Brain.ML.TrainingServerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.TrainingServer

  setup do
    name = :"training_server_test_#{:rand.uniform(100_000)}"
    {:ok, pid} = TrainingServer.start_link(name: name)
    on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)
    {:ok, name: name, pid: pid}
  end

  describe "start_link/1" do
    test "starts with idle status", %{name: name} do
      assert :idle = TrainingServer.get_status(name)
    end
  end

  describe "get_status/1" do
    test "returns :idle when no training", %{name: name} do
      assert :idle = TrainingServer.get_status(name)
    end

    test "returns :idle for unregistered name" do
      assert :idle = TrainingServer.get_status(:nonexistent_training_server)
    end
  end

  describe "start_training/3" do
    test "rejects invalid model type", %{name: name} do
      assert {:error, :invalid_model_type} = TrainingServer.start_training(:invalid, [], name)
    end
  end

  describe "cancel/1" do
    test "returns error when not training", %{name: name} do
      assert {:error, :not_training} = TrainingServer.cancel(name)
    end
  end

  describe "schedule/4" do
    test "creates a schedule", %{name: name} do
      assert {:ok, schedule_id} = TrainingServer.schedule(:tfidf, [], 24, name)
      assert is_binary(schedule_id)
    end
  end

  describe "list_schedules/1" do
    test "returns empty list initially", %{name: name} do
      assert [] = TrainingServer.list_schedules(name)
    end

    test "returns created schedules", %{name: name} do
      {:ok, _id} = TrainingServer.schedule(:tfidf, [], 24, name)
      schedules = TrainingServer.list_schedules(name)
      assert length(schedules) == 1
      [schedule] = schedules
      assert schedule.model_type == :tfidf
      assert schedule.interval_hours == 24
    end

    test "returns empty list for unregistered name" do
      assert [] = TrainingServer.list_schedules(:nonexistent_training_server)
    end
  end

  describe "cancel_schedule/2" do
    test "cancels an existing schedule", %{name: name} do
      {:ok, schedule_id} = TrainingServer.schedule(:unified, [], 12, name)
      assert :ok = TrainingServer.cancel_schedule(schedule_id, name)
      assert [] = TrainingServer.list_schedules(name)
    end

    test "returns error for nonexistent schedule", %{name: name} do
      assert {:error, :not_found} = TrainingServer.cancel_schedule("nonexistent", name)
    end
  end

  describe "handle_info for unknown messages" do
    test "ignores unknown messages", %{pid: pid} do
      send(pid, :some_random_message)
      Process.sleep(10)
      assert Process.alive?(pid)
    end
  end
end
