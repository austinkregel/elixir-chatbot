defmodule Atlas.Settings do
  @moduledoc """
  Context module for node-wide settings (`Atlas.Schemas.Setting`): values
  changed at runtime that must survive a restart, stored by name.
  """

  alias Atlas.Repo
  alias Atlas.Schemas.Setting

  @doc "The stored value of the setting `name`, or `:error` when it has never been written."
  @spec fetch(String.t()) :: {:ok, String.t()} | :error
  def fetch(name) when is_binary(name) do
    case Repo.get(Setting, name) do
      %Setting{value: value} -> {:ok, value}
      nil -> :error
    end
  end

  @doc "Stores `value` as the setting `name`, replacing any earlier value. Raises on a blank name or value."
  @spec put!(String.t(), String.t()) :: Setting.t()
  def put!(name, value) when is_binary(name) and is_binary(value) do
    %Setting{}
    |> Setting.changeset(%{name: name, value: value})
    |> Repo.insert!(
      on_conflict: {:replace, [:value, :updated_at]},
      conflict_target: :name,
      returning: true
    )
  end
end
