defmodule Atlas.Schemas.LexiconFact do
  @moduledoc """
  One fact about one word in the brain's own lexicon.

  The lexicon is seeded from external corpora (WordNet first) and extended by
  what the brain learns. Every fact records where it came from, so a seeded
  fact and a learned fact that contradicts it can coexist; choosing between
  them is done from conversational context at read time.

  ## Fields

  - `word` — the lowercase surface form the fact is about.
  - `kind` — `"sense"`, `"relation"`, or `"property"`.
  - `key` — within a kind: the relation type (`"hypernym"`, `"antonym"`), the
    property name (`"negation"`), or the part of speech of a sense.
  - `ref` — distinguishes several facts sharing word, kind, key and source, such
    as the target of a relation. Empty when there is only one.
  - `value` — the fact's payload.
  - `source` — `"seed:<corpus>"`, `"derived"`, `"clarified"`, or `"authored"`.
  - `confidence`, `frequency`, `archived`, `last_observed_at` — how strongly the
    fact is held and how recently it was seen, for decay.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{}

  @primary_key {:id, :binary_id, autogenerate: true}

  @kinds ~w(sense relation property)
  @plain_sources ~w(derived clarified authored)

  schema "atlas_lexicon_facts" do
    field :word, :string
    field :kind, :string
    field :key, :string
    field :ref, :string, default: ""
    field :value, :map, default: %{}
    field :source, :string
    field :confidence, :float, default: 1.0
    field :frequency, :integer, default: 1
    field :archived, :boolean, default: false
    field :last_observed_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(word kind key source)a
  @optional_fields ~w(ref value confidence frequency archived last_observed_at)a

  @doc "The permitted values of `kind`."
  @spec kinds() :: [String.t()]
  def kinds, do: @kinds

  @doc "The columns that identify a fact. Used as the upsert conflict target."
  @spec identity_fields() :: [atom()]
  def identity_fields, do: [:word, :kind, :key, :ref, :source]

  @doc "Builds a changeset, rejecting anything the lexicon cannot interpret."
  def changeset(fact, attrs) do
    fact
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:kind, @kinds)
    |> validate_normalized(:word)
    |> validate_source()
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:frequency, greater_than_or_equal_to: 0)
    |> validate_ref()
  end

  # Words are stored exactly as looked up. A caller that passes "Dog" or " dog"
  # has a bug, and silently normalising here would hide it.
  defp validate_normalized(changeset, field) do
    validate_change(changeset, field, fn ^field, value ->
      if value == value |> String.trim() |> String.downcase() and value != "" do
        []
      else
        [{field, "must be non-empty, trimmed and lowercase"}]
      end
    end)
  end

  defp validate_source(changeset) do
    validate_change(changeset, :source, fn :source, source ->
      cond do
        source in @plain_sources -> []
        String.starts_with?(source, "seed:") and byte_size(source) > 5 -> []
        true -> [source: "must be seed:<corpus>, derived, clarified or authored"]
      end
    end)
  end

  # An empty ref is valid, but a nil one would bypass the unique index,
  # because Postgres treats NULLs as distinct.
  defp validate_ref(changeset) do
    case get_field(changeset, :ref) do
      nil -> add_error(changeset, :ref, "must not be nil; use an empty string")
      _ -> changeset
    end
  end
end
