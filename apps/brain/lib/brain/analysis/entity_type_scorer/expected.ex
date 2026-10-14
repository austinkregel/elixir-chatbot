defmodule Brain.Analysis.EntityTypeScorer.Expected do
  @moduledoc """
  What a context -- a classified intent or a consolidated domain -- expects of
  the entities in its utterances, as `Brain.Analysis.EntityTypeScorer.expected_types/2`
  and `domain_expected_types/2` build it from `Brain.Analysis.SlotTypePreferences`:

  - `context`: `{:intent, name}` or `{:domain, name}`, the context it was built
    for;
  - `weights`: each entity type the context's slot schemas list, with its
    learned weight (`SlotTypePreferences.weights/3`) -- `%{}` when the schemas
    list none;
  - `readings`: the context's likelihood of every reading of a gazetteer span,
    relative to its likelihood across every utterance
    (`SlotTypePreferences.reading_likelihoods/2`), which the scorer weighs
    each reading by.

  Read it through `types/1`, `weights/1` and `reading_likelihood/2`.
  """

  alias Brain.Analysis.SlotTypePreferences

  @enforce_keys [:context, :weights, :readings]
  defstruct @enforce_keys

  @type t :: %__MODULE__{
          context: {:intent, String.t()} | {:domain, String.t()},
          weights: %{String.t() => float()},
          readings: map()
        }

  @doc "The entity types the context's slot schemas list, sorted."
  @spec types(t()) :: [String.t()]
  def types(%__MODULE__{weights: weights}), do: weights |> Map.keys() |> Enum.sort()

  @doc "Each listed type's learned weight; `%{}` when nothing is listed."
  @spec weights(t()) :: %{String.t() => float()}
  def weights(%__MODULE__{weights: weights}), do: weights

  @doc """
  The context's likelihood of `reading` -- an entity type, or `:ordinary` for
  the ordinary word -- relative to its likelihood across every utterance.
  """
  @spec reading_likelihood(t(), String.t() | :ordinary) :: float()
  def reading_likelihood(%__MODULE__{readings: readings}, reading),
    do: SlotTypePreferences.reading_likelihood(readings, reading)
end
