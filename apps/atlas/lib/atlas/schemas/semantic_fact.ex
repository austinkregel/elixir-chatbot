defmodule Atlas.Schemas.SemanticFact do
  @moduledoc """
  Ecto schema for consolidated semantic facts.

  Replaces the file-based Memory.Store semantic fact persistence.
  Semantic facts are consolidated from episode clusters and represent
  generalized knowledge with confidence scores and TF-IDF embeddings.

  `content` is the fact's own text: for a fact consolidated from episodes, the
  examples of what the user said (the episodes' states, joined by `" | "`).
  `action` is the action those episodes most often had; a fact that was not
  consolidated from episodes has none.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  # The prefix and separator of the single string consolidation once stored a
  # fact as: "Action: <action> | Examples: <examples>".
  @legacy_action_prefix "Action: "
  @legacy_examples_separator " | Examples: "

  schema "atlas_semantic_facts" do
    field :world_id, :string
    field :content, :string
    field :action, :string
    field :category, :string
    field :confidence, :float
    field :embedding, {:array, :float}, default: []
    field :source_episodes, {:array, :string}, default: []
    field :tags, {:array, :string}, default: []

    timestamps(type: :utc_datetime_usec)
  end

  # A semantic fact belongs to the world its episodes came from; the caller
  # names it.
  @required_fields ~w(world_id content category confidence)a
  @optional_fields ~w(action embedding source_episodes tags)a

  def changeset(semantic_fact, attrs) do
    semantic_fact
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  @doc """
  Splits a fact stored in the single string consolidation once wrote,
  `"Action: <action> | Examples: <examples>"`, into `{action, examples}`.

  Text not in that form is a fact's own text with no action: `{nil, text}`.
  The migration that added the `action` column converts the rows already
  stored the same way (`priv/repo/migrations/20261009180000_add_action_to_semantic_facts.exs`);
  this converts what is read from the files written before it.
  """
  @spec split_legacy_content(String.t()) :: {String.t() | nil, String.t()}
  def split_legacy_content(@legacy_action_prefix <> rest = content) do
    case String.split(rest, @legacy_examples_separator, parts: 2) do
      [action, examples] -> {action, examples}
      [_no_examples] -> {nil, content}
    end
  end

  def split_legacy_content(content) when is_binary(content), do: {nil, content}

  @doc "Query semantic facts for a specific world."
  def for_world(query \\ __MODULE__, world_id) do
    from(sf in query, where: sf.world_id == ^world_id)
  end

  @doc "Query semantic facts by category."
  def for_category(query \\ __MODULE__, category) do
    from(sf in query, where: sf.category == ^category)
  end
end
