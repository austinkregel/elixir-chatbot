defmodule Atlas.Schemas.Credential do
  @moduledoc """
  Ecto schema for encrypted credential storage.

  Replaces the file-based CredentialVault `.enc` persistence.
  Credentials are scoped by world and service, with the actual
  values encrypted at the application layer before storage.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_credentials" do
    field :world, :string, default: "default"
    field :service, :string
    field :key, :string
    field :encrypted_value, :binary

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(service key encrypted_value)a
  @optional_fields ~w(world)a

  def changeset(credential, attrs) do
    credential
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint([:world, :service, :key])
  end

  @doc "Query credentials scoped to a world."
  def for_world(query \\ __MODULE__, world) do
    from(c in query, where: c.world == ^world)
  end

  @doc "Query credentials for a specific service within a world."
  def for_service(query \\ __MODULE__, service) do
    from(c in query, where: c.service == ^service)
  end
end
