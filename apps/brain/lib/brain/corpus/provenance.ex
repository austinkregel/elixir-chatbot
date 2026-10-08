defmodule Brain.Corpus.Provenance do
  @moduledoc """
  Where a row in `data/intents/*_usersays_en.json` came from.

  The export is not purely Dialogflow output. `scripts/materialize_orphan_intents.exs`
  writes gold-standard texts that have no source backing *back into* those files,
  tagging them `orphan-<hash>` or `augmented-<hash>`. Materialized text reads as a
  plausible utterance; augmented text is a mutation of a real one and can be
  ungrammatical. Accuracy measured on either is not accuracy on user input, so the
  distinction has to survive into the corpus rather than being inferred later from
  an id prefix by whoever happens to need it.

  Every consumer reads the kind from here, so there is one answer to what
  "augmented" covers.
  """

  # Ordered most genuine first. The integer is the comparison key, not an id.
  @ranks %{"dialogflow" => 0, "materialized" => 1, "augmented" => 2}

  @synthetic_id ~r/^[a-z]+-[0-9a-f]{8}$/

  @doc "Every kind, most genuine first."
  @spec kinds() :: [String.t()]
  def kinds, do: @ranks |> Map.keys() |> Enum.sort_by(&Map.fetch!(@ranks, &1))

  @doc "True when `kind` is one this module recognises."
  @spec kind?(term()) :: boolean()
  def kind?(kind), do: Map.has_key?(@ranks, kind)

  @doc "The comparison key for a kind. Raises on an unknown one."
  @spec rank!(String.t()) :: non_neg_integer()
  def rank!(kind), do: Map.fetch!(@ranks, kind)

  @doc """
  The kind of a usersays row's `id`.

  A missing id means Dialogflow rather than unknown: the synthetic writers always
  set one, while the export omits it on a few otherwise ordinary entries.

  Raises on an id shaped like a synthetic marker whose prefix is not known, since
  such a row's origin cannot be recorded and guessing it is how the corpus came to
  claim every row was Dialogflow's.
  """
  @spec of_id!(term(), Path.t()) :: String.t()
  def of_id!(nil, _path), do: "dialogflow"

  def of_id!("orphan-" <> _, _path), do: "materialized"
  def of_id!("augmented-" <> _, _path), do: "augmented"

  def of_id!(id, path) when is_binary(id) do
    if Regex.match?(@synthetic_id, id) do
      raise """
      unrecognised synthetic id prefix in #{path}

        id: #{id}

      `orphan-` and `augmented-` are written by scripts/materialize_orphan_intents.exs.
      This prefix is neither, so the row's origin is unknown. Teach
      Brain.Corpus.Provenance.of_id!/2 what writes it.
      """
    else
      "dialogflow"
    end
  end

  def of_id!(other, path) do
    raise "#{path} has a non-string id #{inspect(other)}"
  end

  @doc """
  The kind of a usersays entry, read from its `id`.
  """
  @spec of_entry!(map(), Path.t()) :: String.t()
  def of_entry!(entry, path) when is_map(entry), do: of_id!(Map.get(entry, "id"), path)

  @doc """
  The most genuine kind among several.

  The same normalised text can appear as a genuine phrase in one file and as a
  materialized copy in another. A text is synthetic only when it exists nowhere
  outside a synthetic writer, so the most genuine origin wins.
  """
  @spec most_genuine([String.t()]) :: String.t()
  def most_genuine([_ | _] = kinds), do: Enum.min_by(kinds, &rank!/1)
end
