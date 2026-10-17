defmodule Brain.ML.TrainingServerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.TrainingServer

  setup_all do
    unless Process.whereis(Brain.PubSub) do
      start_supervised!({Phoenix.PubSub, name: Brain.PubSub})
    end

    :ok
  end

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    name = :"training_server_test_#{:rand.uniform(100_000)}"
    runs_root = Path.join(tmp_dir, "runs")
    {:ok, pid} = TrainingServer.start_link(name: name, runs_root: runs_root)
    on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)
    {:ok, name: name, pid: pid, runs_root: runs_root}
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

    test "exits for an unregistered name rather than reporting idle" do
      assert {:noproc, _} = catch_exit(TrainingServer.get_status(:nonexistent_training_server))
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

    test "exits for an unregistered name rather than reporting no schedules" do
      assert {:noproc, _} = catch_exit(TrainingServer.list_schedules(:nonexistent_training_server))
    end
  end

  describe "cancel_schedule/2" do
    test "cancels an existing schedule", %{name: name} do
      {:ok, schedule_id} = TrainingServer.schedule(:tfidf, [], 12, name)
      assert :ok = TrainingServer.cancel_schedule(schedule_id, name)
      assert [] = TrainingServer.list_schedules(name)
    end

    test "returns error for nonexistent schedule", %{name: name} do
      assert {:error, :not_found} = TrainingServer.cancel_schedule("nonexistent", name)
    end
  end

  # Small on purpose: these check the job plumbing (progress, recording,
  # cancel), not the model. 40 sentences for a few epochs.
  describe "a POS run" do
    setup do
      Phoenix.PubSub.subscribe(Brain.PubSub, TrainingServer.pos_topic())
      :ok
    end

    test "refuses invalid params before starting", %{name: name} do
      assert {:error, message} = TrainingServer.start_training(:pos, [sentences: 40, max_epochs: 0, patience: nil, seed: 1], name)
      assert message =~ "max_epochs"
      assert :idle = TrainingServer.get_status(name)
    end

    test "broadcasts every epoch and records the finished run", %{name: name, runs_root: root} do
      assert {:ok, :pos, id} = TrainingServer.start_training(:pos, [sentences: 40, max_epochs: 2, patience: nil, seed: 1], name)
      assert TrainingServer.current_run(name) == id

      assert_receive {:pos_run_started, ^id}
      assert_receive {:pos_epoch, ^id, %{epoch: 1, loss: loss}}, 120_000
      assert is_float(loss) and loss > 0
      assert_receive {:pos_epoch, ^id, %{epoch: 2}}, 120_000
      assert_receive {:pos_run_finished, ^id, "completed"}, 120_000

      run = Brain.Training.POSRuns.get!(id, root)
      assert run["status"] == "completed"
      assert Enum.map(run["curve"], & &1["epoch"]) == [1, 2]
      assert "epoch-2" in run["snapshots"]
      assert :idle = TrainingServer.get_status(name)
    end

    test "cancel kills the job and records the run as cancelled", %{name: name, runs_root: root} do
      assert {:ok, :pos, id} = TrainingServer.start_training(:pos, [sentences: 40, max_epochs: 500, patience: nil, seed: 1], name)
      assert_receive {:pos_epoch, ^id, %{epoch: 1}}, 120_000

      :sys.get_state(name) |> Map.fetch!(:task) |> Map.fetch!(:pid) |> Process.monitor()
      assert :ok = TrainingServer.cancel(name)
      assert_receive {:DOWN, _, :process, _, _reason}, 5_000

      assert Brain.Training.POSRuns.get!(id, root)["status"] == "cancelled"
      assert_receive {:pos_run_finished, ^id, "cancelled"}
      assert :idle = TrainingServer.get_status(name)
    end

    test "a run left running by an earlier server is marked interrupted at start", %{runs_root: root} do
      dir = Path.join(root, "stale")
      File.mkdir_p!(dir)
      File.write!(Path.join(dir, "run.json"), Jason.encode!(%{"id" => "stale", "status" => "running"}))

      {:ok, pid} = TrainingServer.start_link(name: :training_server_restart_test, runs_root: root)
      GenServer.stop(pid)

      assert Brain.Training.POSRuns.get!("stale", root)["status"] == "interrupted"
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
