defmodule Atlas.Schemas.WebUser do
  @moduledoc """
  A person using the web interface, identified by the id their browser keeps in
  its signed session cookie, and the world they have selected.

  The selected world is where that user's new conversations run: the web
  interface passes it to `Brain.create_conversation/1`.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @type t :: %__MODULE__{}

  schema "web_users" do
    field :selected_world_id, :string

    timestamps(type: :utc_datetime_usec)
  end

  @doc false
  def changeset(web_user, attrs) do
    web_user
    |> cast(attrs, [:selected_world_id])
    |> validate_required([:selected_world_id])
    |> validate_change(:selected_world_id, fn :selected_world_id, value ->
      if String.trim(value) == "", do: [selected_world_id: "must not be blank"], else: []
    end)
  end
end
