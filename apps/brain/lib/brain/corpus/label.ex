defmodule Brain.Corpus.Label do
  @moduledoc """
  How the corpus names an intent.

  One export file's utterances are filed under one label, and two transformations
  stand between the filename and that label:

    * everything from the first `" - "` onward is a *context* on the intent, not a
      separate intent — `account.balance.check - context_ account` is
      `account.balance.check` asked as a follow-up;
    * `smarthome.lights.*` is a retired namespace that folds onto
      `smarthome.device.*`.

  Anything that compares itself to the corpus — an accuracy figure, a coverage
  report, a registry — has to name intents this way, or it is comparing two
  different vocabularies and the mismatch looks like missing data.

  ## The annotated mirror

  `data/training/intents/` names the same intents a third way, because
  `scripts/generate_pos_annotations.py` flattens `" - "` and spaces to dots:
  `account.balance.check.context_.account`.

  That mapping is only decidable forwards. A `.` in a flattened name may be a real
  dot or a flattened separator, and nothing in the name says which, so `mangle/1`
  runs *from* canonical names and `annotated_index/1` builds the lookup in that
  direction. Reversing it would be guessing.
  """

  @lights_prefix "smarthome.lights."
  @device_prefix "smarthome.device."

  @separator " - "

  @doc """
  The intent an export label belongs to, with its context suffix dropped.

      iex> Brain.Corpus.Label.root("account.balance.check - context_ account")
      "account.balance.check"
  """
  @spec root(String.t()) :: String.t()
  def root(name) when is_binary(name) do
    name |> String.split(@separator) |> List.first() |> String.trim()
  end

  @doc """
  Folds the retired `smarthome.lights.` namespace onto `smarthome.device.`.
  """
  @spec fold_lights(String.t()) :: String.t()
  def fold_lights(@lights_prefix <> rest), do: @device_prefix <> rest
  def fold_lights(label) when is_binary(label), do: label

  @doc """
  The label the corpus files an export file's utterances under.

  This is the vocabulary `Pipeline.registered_intent?/1` gates accuracy on, so
  anything comparing itself to the corpus has to name intents this way or it is
  comparing two different vocabularies.
  """
  @spec canonical(String.t()) :: String.t()
  def canonical(name) when is_binary(name), do: name |> root() |> fold_lights()

  @doc """
  The stem `scripts/generate_pos_annotations.py` derives from an export stem.

  Its `extract_intent_name` does `name.replace(' - ', '.').replace(' ', '.')`
  after stripping the suffix.
  """
  @spec mangle(String.t()) :: String.t()
  def mangle(stem) when is_binary(stem) do
    stem |> String.replace(@separator, ".") |> String.replace(" ", ".")
  end

  @doc """
  Maps each annotated-corpus stem to the canonical label it belongs to.

  Built forward from the export's own stems, which is the only direction that is
  decidable. A stem the export cannot account for is absent from the result, and
  callers are expected to keep such a name as it stands rather than invent a
  canonical form for it.
  """
  @spec annotated_index([String.t()]) :: %{optional(String.t()) => String.t()}
  def annotated_index(export_stems) when is_list(export_stems) do
    Map.new(export_stems, fn stem -> {mangle(stem), canonical(stem)} end)
  end
end
