defmodule Fleet.WorkspaceTestBackend do
  @moduledoc """
  A minimal in-memory holodeck backend for Fleet's workspace-tool tests.

  Fleet depends on `fourth_wall` as a compiled library, so it cannot reach that
  app's own test-support `Backend.Memory`. This is the same idea, scoped to
  Fleet's suite: it lets the workspace tool handlers be exercised end-to-end —
  proving they key on the officer's own soul — without a Docker daemon.
  """

  @behaviour FourthWall.Holodeck.Backend

  @table :fleet_workspace_test_backend

  @impl true
  def create(soul_id, _opts) when is_binary(soul_id) do
    ensure_table()
    {:ok, %{soul_id: soul_id}}
  end

  @impl true
  def destroy(%{soul_id: soul_id}) do
    ensure_table()
    :ets.match_delete(@table, {{soul_id, :_}, :_})
    :ok
  end

  @impl true
  def read(%{soul_id: soul_id}, abs_path) do
    ensure_table()

    case :ets.lookup(@table, {soul_id, abs_path}) do
      [{_, content}] -> {:ok, content}
      [] -> {:error, {:read_failed, "no such file"}}
    end
  end

  @impl true
  def write(%{soul_id: soul_id}, abs_path, content) do
    ensure_table()
    :ets.insert(@table, {{soul_id, abs_path}, content})
    :ok
  end

  @impl true
  def list(%{soul_id: soul_id}, _abs_path) do
    ensure_table()
    paths = :ets.match(@table, {{soul_id, :"$1"}, :_}) |> List.flatten() |> Enum.sort()
    {:ok, Enum.join(paths, "\n")}
  end

  @impl true
  def exec(%{soul_id: _}, argv), do: {:ok, %{stdout: Enum.join(argv, " "), exit_status: 0}}

  defp ensure_table do
    case :ets.whereis(@table) do
      :undefined -> :ets.new(@table, [:public, :named_table, :set])
      _ -> :ok
    end
  rescue
    ArgumentError -> :ok
  end
end
