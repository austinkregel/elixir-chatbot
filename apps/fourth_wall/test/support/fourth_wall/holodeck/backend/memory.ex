defmodule FourthWall.Holodeck.Backend.Memory do
  @moduledoc """
  An in-memory holodeck backend for tests.

  Real enough to exercise `FourthWall.Holodeck`'s lifecycle, per-soul keying, and
  the workspace tool handlers — creation is idempotent, reads see what writes
  wrote, and destroy reclaims a soul's files — without a Docker daemon. Files
  live in a public ETS table keyed by soul id, so distinct souls are genuinely
  isolated and a test can assert one workspace cannot see another's.

  Not a boundary: it runs nothing and confines nothing. It exists so the code
  *above* the boundary can be tested where the boundary itself cannot be.
  """

  @behaviour FourthWall.Holodeck.Backend

  @table :holodeck_memory_backend

  @impl true
  def create(soul_id, _opts) when is_binary(soul_id) do
    ensure_table()
    :ets.insert(@table, {{soul_id, :exists}, true})
    {:ok, %{soul_id: soul_id}}
  end

  @impl true
  def destroy(%{soul_id: soul_id}) do
    ensure_table()
    :ets.match_delete(@table, {{soul_id, :_, :_}, :_})
    :ets.match_delete(@table, {{soul_id, :_}, :_})
    :ok
  end

  @impl true
  def read(%{soul_id: soul_id}, abs_path) do
    ensure_table()

    case :ets.lookup(@table, {soul_id, :file, abs_path}) do
      [{_, content}] -> {:ok, content}
      [] -> {:error, {:read_failed, "no such file: #{abs_path}"}}
    end
  end

  @impl true
  def write(%{soul_id: soul_id}, abs_path, content) when is_binary(content) do
    ensure_table()
    :ets.insert(@table, {{soul_id, :file, abs_path}, content})
    :ok
  end

  @impl true
  def list(%{soul_id: soul_id}, abs_path) do
    ensure_table()
    prefix = String.trim_trailing(abs_path, "/") <> "/"

    entries =
      :ets.match(@table, {{soul_id, :file, :"$1"}, :_})
      |> List.flatten()
      |> Enum.filter(&(&1 == abs_path or String.starts_with?(&1, prefix)))
      |> Enum.sort()

    {:ok, Enum.join(entries, "\n")}
  end

  @impl true
  def exec(%{soul_id: _soul_id}, argv) when is_list(argv) do
    {:ok, %{stdout: Enum.join(argv, " "), exit_status: 0}}
  end

  defp ensure_table do
    case :ets.whereis(@table) do
      :undefined ->
        :ets.new(@table, [:public, :named_table, :set])

      _ ->
        :ok
    end
  rescue
    ArgumentError -> :ok
  end
end
