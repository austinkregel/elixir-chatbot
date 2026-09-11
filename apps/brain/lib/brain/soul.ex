defmodule Brain.Soul do
  @moduledoc """
  A Soul: a portable, per-agent constitutional document — identity, values, and
  behavioral bounds — that shapes how an agent generates.

  Souls are their own persisted entities (files on disk), NOT owned by a World.
  A World holds a *roster of residents* (soul ids); a Soul can reside in more
  than one World, which is what lets the same Soul be run under different world
  framings and compared. See `World.TrainingWorld` `:residents`.

  This is the promotion of the (previously dead) single global `persona` into
  per-agent Souls. `system_prompt/1` renders the Soul's constitution as the
  system message that `Brain.Response.RealizationPacket` sends to the model.
  """

  @type id :: String.t()
  @type t :: %__MODULE__{
          id: id(),
          name: String.t(),
          constitution: String.t() | nil,
          genome: map(),
          metadata: map()
        }

  @enforce_keys [:id]
  defstruct [:id, :name, :constitution, genome: %{}, metadata: %{}]

  @doc """
  Directory Souls are loaded from. Plain files, inspectable and greppable
  (defaults to `<umbrella_root>/souls`); override with
  `config :brain, :souls_dir, "/some/path"`.
  """
  def souls_dir do
    Application.get_env(:brain, :souls_dir, Path.join(File.cwd!(), "souls"))
  end

  @doc "Loads a Soul by id from `souls_dir/<id>.json`. Returns `{:ok, soul}` or `{:error, reason}`."
  @spec get(id()) :: {:ok, t()} | {:error, term()}
  def get(id) when is_binary(id) do
    path = Path.join(souls_dir(), "#{id}.json")

    with {:ok, raw} <- File.read(path),
         {:ok, data} <- Jason.decode(raw) do
      {:ok, from_map(data)}
    else
      {:error, reason} -> {:error, reason}
      _ -> {:error, :invalid_soul}
    end
  end

  @doc "Lists the ids of all Souls on file."
  def list_ids do
    case File.ls(souls_dir()) do
      {:ok, files} ->
        files |> Enum.filter(&String.ends_with?(&1, ".json")) |> Enum.map(&Path.rootname/1)

      _ ->
        []
    end
  end

  @doc """
  Renders the Soul's system prompt — the constitution text that defines who the
  agent is. Returns `nil` if the Soul has no constitution (callers fall back to
  the generic default).
  """
  @spec system_prompt(t()) :: String.t() | nil
  def system_prompt(%__MODULE__{constitution: c}) when is_binary(c) and c != "", do: c
  def system_prompt(_), do: nil

  defp from_map(data) when is_map(data) do
    %__MODULE__{
      id: data["id"],
      name: data["name"],
      constitution: data["constitution"],
      genome: data["genome"] || %{},
      metadata: data["metadata"] || %{}
    }
  end
end
