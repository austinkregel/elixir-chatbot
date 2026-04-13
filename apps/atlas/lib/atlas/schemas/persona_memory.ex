defmodule Atlas.Schemas.PersonaMemory do
  @moduledoc """
  Ecto schema for legacy persona memory storage.

  Replaces the file-based MemoryStore per-persona .json persistence.
  Stores message arrays from legacy persona memory files.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_persona_memories" do
    field :persona_name, :string
    field :role, :string
    field :content, :string
    field :context, :map, default: %{}
    field :message_timestamp, :integer

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(persona_name role content)a
  @optional_fields ~w(context message_timestamp)a

  def changeset(persona_memory, attrs) do
    persona_memory
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
  end

  @doc "Query memories for a persona."
  def for_persona(query \\ __MODULE__, persona_name) do
    from(pm in query, where: pm.persona_name == ^persona_name)
  end
end
