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

  ## Authored examples

  Intent examples written by hand as examples of their intent, which have no
  Dialogflow export to come from, are kept outside the export in
  `priv/training/authored_intents/` (`authored_dir/0`), in the export's
  usersays format. Their kind is `"authored"`, read from the directory they
  are in rather than from an id. They were never in the export, so they are
  not among the kinds written back into it (`written_back/0`).

  ## Authored entity annotations

  Entity annotations written by hand for intent gold texts the export
  annotates no entity in are kept in `priv/training/authored_entities/`
  (`authored_entities_dir/0`), one usersays file per intent, each entity a
  `data` segment marked as Dialogflow marks one: `meta` names its entity type
  (`"@function"`), `alias` the intent's slot it fills (`"symbol"`). A segment
  whose `meta` is `"@sys.any"` marks the value of a free-text slot
  (`"reminder_text"`) instead, which names no entity. They are the source of
  the NER gold rows labeled `"authored"`, which `mix rebuild_ner_gold`
  rebuilds from them, and, with the authored intent examples, of the spans
  `Brain.Analysis.FreeTextCapture` learns from. Only the annotations are
  authored: the texts are intent gold rows and keep the provenance the intent
  gold records for them.
  """

  # Ordered most genuine first. The integer is the comparison key, not an id.
  # An authored example is written as an example of its intent; a materialized
  # row is a gold text that had no source, written back into the export.
  @ranks %{"dialogflow" => 0, "authored" => 1, "materialized" => 2, "augmented" => 3}

  # The id prefix each synthetic writer marks the rows it writes back into the
  # export with, and the kind it marks.
  @written_back_prefixes %{"orphan-" => "materialized", "augmented-" => "augmented"}

  @authored "authored"
  @authored_dir "training/authored_intents"
  @authored_entities_dir "training/authored_entities"

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
  The kinds a synthetic writer marks on the rows it writes back into the
  export, most genuine first: the rows of the export that did not come from
  Dialogflow.
  """
  @spec written_back() :: [String.t()]
  def written_back, do: @written_back_prefixes |> Map.values() |> Enum.sort_by(&rank!/1)

  @doc "The kind of every hand-written intent example in `authored_dir/0`."
  @spec authored() :: String.t()
  def authored, do: @authored

  @doc "Where the hand-written intent examples are kept, one usersays file per intent."
  @spec authored_dir() :: Path.t()
  def authored_dir, do: Brain.priv_path(@authored_dir)

  @doc """
  Where the hand-written entity annotations of intent gold texts are kept, one
  usersays file per intent. The NER gold rows built from them are labeled
  `authored/0`.
  """
  @spec authored_entities_dir() :: Path.t()
  def authored_entities_dir, do: Brain.priv_path(@authored_entities_dir)

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

  def of_id!(id, path) when is_binary(id) do
    case Enum.find(@written_back_prefixes, fn {prefix, _kind} -> String.starts_with?(id, prefix) end) do
      {_prefix, kind} -> kind
      nil -> unmarked_id!(id, path)
    end
  end

  def of_id!(other, path) do
    raise "#{path} has a non-string id #{inspect(other)}"
  end

  defp unmarked_id!(id, path) do
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
