defmodule World.EntityCandidate do
  @moduledoc """
  One observation of an entity a world has not yet promoted to its gazetteer.

  A candidate is not a discovery result. `World.EntityDiscoverer`'s
  `discovery_result` describes what a single mention looked like in one text --
  its position, the gazetteer types already known for it, and a `:status` of
  `:known`, `:ambiguous` or `:unknown`. A candidate is the narrower thing the
  promoter accumulates: a value, the type something inferred for it, how
  confident that was, and the text it was seen in.

  ## Why this exists

  Candidates used to be plain maps, built independently by five producers that
  each set a different subset of keys. `World.Manager.add_candidate/2` guarded
  only `is_map/1`, so nothing checked them. `World.EntityPromoter` then fed them
  to `EntityDiscoverer.aggregate_discoveries/1` -- written for discovery
  results -- which read `&1.context` and killed the promoter with a `KeyError`
  on every scan that saw a candidate from the type-narrowing path. Alongside it,
  `Map.get(&1, :status) || :unknown` silently passed every candidate through a
  filter meant to select unknown discoveries, because no candidate has a
  `:status` at all. Measured 2026-09-23; the crash fired three times in a single
  dev run.

  `:context` is required because it is what a human reviewing a suggestion
  actually reads -- `World.EntityPromoter.review_candidate/2` joins it into the
  finding's `raw_context`. A candidate without it produces a suggestion nobody
  can judge.
  """

  # Candidates round-trip through `discovered_entities.json` per world
  # (World.Persistence), so the struct has to be encodable.
  @derive Jason.Encoder

  @enforce_keys [:value, :inferred_type, :confidence, :context]
  defstruct [
    :value,
    :inferred_type,
    :confidence,
    :context,
    :source,
    :original_type,
    :discovered_at,
    occurrences: 1,
    metadata: %{}
  ]

  @type t :: %__MODULE__{
          value: String.t(),
          inferred_type: String.t(),
          confidence: float(),
          context: String.t(),
          source: atom() | String.t() | nil,
          original_type: String.t() | nil,
          discovered_at: DateTime.t() | nil,
          occurrences: pos_integer(),
          metadata: map()
        }

  @required @enforce_keys

  @doc """
  Builds a candidate from an attribute map, raising if it is not one.

  Accepts a map or keyword list with atom keys. `:discovered_at` defaults to
  now. Raises `ArgumentError` naming every missing or empty required field --
  producers live in three different umbrella apps, so the message has to say
  which one is at fault.
  """
  @spec new!(map() | keyword()) :: t()
  def new!(attrs) when is_list(attrs), do: attrs |> Map.new() |> new!()

  def new!(%__MODULE__{} = candidate), do: candidate

  def new!(attrs) when is_map(attrs) do
    missing =
      Enum.filter(@required, fn key ->
        case Map.get(attrs, key) do
          nil -> true
          "" -> true
          _ -> false
        end
      end)

    unless missing == [] do
      raise ArgumentError, """
      World.EntityCandidate: missing #{inspect(missing)}

      Got: #{inspect(attrs, limit: 12)}

      Every candidate needs #{inspect(@required)}. :context is the text the
      entity was observed in; EntityPromoter shows it to whoever reviews the
      suggestion, so a candidate without it cannot be judged.
      """
    end

    struct!(
      __MODULE__,
      attrs
      |> Map.take([
        :value,
        :inferred_type,
        :confidence,
        :context,
        :source,
        :original_type,
        :discovered_at,
        :occurrences,
        :metadata
      ])
      |> Map.put_new(:discovered_at, DateTime.utc_now())
    )
  end

  @doc "The keys a candidate must carry."
  @spec required_keys() :: [atom()]
  def required_keys, do: @required
end
