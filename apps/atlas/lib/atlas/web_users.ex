defmodule Atlas.WebUsers do
  @moduledoc """
  Context module for the web interface's users (`Atlas.Schemas.WebUser`).

  A user is created on a browser's first visit and found again by the id its
  session cookie carries. Whether a world exists is the world app's to say, so
  callers check that before selecting one; this module only stores the choice.
  """

  alias Atlas.Repo
  alias Atlas.Schemas.WebUser

  @doc "Creates a user whose conversations start in `world_id`. Raises on a blank world."
  @spec create!(String.t()) :: WebUser.t()
  def create!(world_id) when is_binary(world_id) do
    %WebUser{}
    |> WebUser.changeset(%{selected_world_id: world_id})
    |> Repo.insert!()
  end

  @doc """
  The user with this id, or raises. A session that names a user with no row
  means the row was deleted underneath it; continuing as someone else would
  attach this person's conversations to the wrong user.
  """
  @spec get!(Ecto.UUID.t()) :: WebUser.t()
  def get!(id) when is_binary(id), do: Repo.get!(WebUser, id)

  @doc """
  The PubSub topic of the user's open pages: what is said there reaches every
  page the user has open (`ChatWeb.WorldContext` subscribes each one).
  """
  @spec topic(Ecto.UUID.t()) :: String.t()
  def topic(id) when is_binary(id), do: "web_user:#{id}"

  @doc "Stores `world_id` as the user's selected world. Raises on a blank world."
  @spec select_world!(WebUser.t(), String.t()) :: WebUser.t()
  def select_world!(%WebUser{} = user, world_id) when is_binary(world_id) do
    user
    |> WebUser.changeset(%{selected_world_id: world_id})
    |> Repo.update!()
  end
end
