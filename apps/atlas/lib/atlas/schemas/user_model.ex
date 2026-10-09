defmodule Atlas.Schemas.UserModel do
  @moduledoc """
  Ecto schema for user model storage.

  A user model is what the system holds about one web user
  (`Atlas.Schemas.WebUser`): facts, interaction patterns, epistemic bounds,
  provenance and disclosure history. It has no world. The same model is used
  in every world the user talks in; a fact is put into a world's beliefs only
  when it is synced there.

  `user_id` is the web user's id, a foreign key to `web_users`, and unique: a
  web user has at most one user model.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_user_models" do
    # Ecto.UUID, not :binary_id: a value that is no uuid is refused when cast,
    # as a changeset error, instead of raising when the row is written.
    belongs_to :web_user, Atlas.Schemas.WebUser, foreign_key: :user_id, type: Ecto.UUID
    field :facts, :map, default: %{}
    field :interaction_patterns, :map, default: %{}
    field :epistemic_bounds, :map, default: %{}
    field :provenance_map, :map, default: %{}
    field :disclosure_history, {:array, :map}, default: []

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(user_id)a
  @optional_fields ~w(facts interaction_patterns epistemic_bounds provenance_map disclosure_history)a

  def changeset(user_model, attrs) do
    user_model
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint(:user_id)
    |> foreign_key_constraint(:user_id)
  end

  @doc "Query by user ID."
  def for_user(query \\ __MODULE__, user_id) do
    from(um in query, where: um.user_id == ^user_id)
  end
end
