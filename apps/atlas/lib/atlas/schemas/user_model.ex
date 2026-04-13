defmodule Atlas.Schemas.UserModel do
  @moduledoc """
  Ecto schema for user model storage.

  Replaces the file-based user_models.term persistence.
  Stores per-user facts, interaction patterns, and epistemic bounds.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_user_models" do
    field :user_id, :string
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
  end

  @doc "Query by user ID."
  def for_user(query \\ __MODULE__, user_id) do
    from(um in query, where: um.user_id == ^user_id)
  end
end
