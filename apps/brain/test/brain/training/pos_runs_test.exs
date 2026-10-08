defmodule Brain.Training.POSRunsTest do
  @moduledoc """
  The run recorder's plumbing on a tiny run: what it writes, when it
  refuses. Model quality is measured on the promoted test model instead
  (`Brain.ML.POSTaggerTest`).
  """
  use ExUnit.Case, async: false

  alias Brain.Training.POSRuns

  @moduletag :tmp_dir

  @tiny %{sentences: 40, max_epochs: 5, patience: nil, seed: 3}

  describe "run!/2" do
    test "records settings, provenance, the curve and snapshots", %{tmp_dir: root} do
      run = POSRuns.run!(@tiny, root: root, id: "tiny")

      assert run["status"] == "completed"
      assert run["params"] == %{"sentences" => 40, "max_epochs" => 5, "patience" => nil, "seed" => 3}
      assert Enum.map(run["inputs"], & &1["name"]) == ~w(ud_ewt.train.json ud_ewt.dev.json ud_ewt.test.json)
      assert Enum.map(run["curve"], & &1["epoch"]) == [1, 2, 3, 4, 5]
      assert Enum.all?(run["curve"], &(is_float(&1["loss"]) and is_float(&1["dev_accuracy"])))

      # 5 is a snapshot epoch and the last one; best is whichever epoch led on dev.
      assert "best" in run["snapshots"] and "epoch-5" in run["snapshots"]
      assert run["best"]["epoch"] in 1..5

      for snapshot <- run["snapshots"],
          do: assert(File.exists?(Path.join([root, "tiny", snapshot <> ".term"])))

      assert POSRuns.list(root) == [run]
    end

    test "refuses bad params before creating anything", %{tmp_dir: root} do
      assert_raise ArgumentError, ~r/max_epochs/, fn -> POSRuns.run!(%{@tiny | max_epochs: 0}, root: root) end
      assert_raise ArgumentError, ~r/unknown param/, fn -> POSRuns.run!(Map.put(@tiny, :epochs, 5), root: root) end
      assert POSRuns.list(root) == []
    end
  end

  describe "promote!/4" do
    test "refuses a snapshot that does not beat the lookup baseline, saving nothing", %{tmp_dir: root} do
      # Five epochs on 40 sentences cannot beat a lookup table.
      POSRuns.run!(@tiny, root: root, id: "weak")
      out = Path.join(root, "promoted.term")

      assert_raise RuntimeError, ~r/does not beat the most-frequent-tag baseline/, fn ->
        POSRuns.promote!("weak", "best", :test, root: root, to: out)
      end

      refute File.exists?(out)
      assert POSRuns.get!("weak", root)["promotions"] == []
    end

    test "refuses a snapshot the run does not have", %{tmp_dir: root} do
      POSRuns.run!(%{@tiny | max_epochs: 1}, root: root, id: "short")

      assert_raise ArgumentError, ~r/no snapshot "epoch-50"/, fn ->
        POSRuns.promote!("short", "epoch-50", :test, root: root)
      end
    end
  end

  describe "status" do
    test "a run still running when the server starts is marked interrupted", %{tmp_dir: root} do
      POSRuns.run!(%{@tiny | max_epochs: 1}, root: root, id: "done")
      File.mkdir_p!(Path.join(root, "stale"))
      File.write!(Path.join([root, "stale", "run.json"]), Jason.encode!(%{"id" => "stale", "status" => "running"}))

      assert POSRuns.mark_interrupted!(root) == ["stale"]
      assert POSRuns.get!("stale", root)["status"] == "interrupted"
      assert POSRuns.get!("done", root)["status"] == "completed"
    end
  end
end
