defmodule Brain.Analysis.SpeechActTable do
  @moduledoc """
  The speech act of each intent: its category and sub-type, used when the
  intent classifier's answer refines a chunk's speech act.

  The intent registry (`priv/analysis/intent_registry.json`) declares a
  `"category"` for every intent and a `"speech_act"` sub-type for most. An
  intent whose sub-type is not declared gets one learned from the data: a
  `Brain.Analysis.LexicalEvidence` model is fitted on the training utterances of
  the intents that do declare one, labeled with their category and sub-type,
  its prior strength chosen as `LexicalEvidence.select/2` chooses one, and each
  undeclared intent takes the sub-type most of its own training utterances are
  read as. Each entry records whether its sub-type was declared or learned.

  This replaces a list of intent-name prefixes in the speech-act classifier,
  which named domains ("greeting", "farewell") that no intent in the registry
  uses, so every greeting intent read as an assertive statement.

  ## What the words say about the speech act

  The table keeps that model as `evidence`: its labels are
  `"<category>:<sub_type>"`, one for each declared pair, so it reads an
  utterance's words as evidence for a speech act pooled across every intent
  that declares it. An intent's own training rows can be few (a greeting
  intent has under twenty), but the greetings of every greeting intent
  together are many, so a one-word "Hello" is read as a greeting from the
  data. `Brain.Analysis.SpeechActClassifier` votes with it in pass 1, before
  any intent is known.

  Built per language at training time (`speech_act_table.<language>.term`,
  directly under the models directory) and loaded at boot.
  """

  alias Brain.Analysis.LexicalEvidence

  @enforce_keys [:language, :entries, :evidence]
  defstruct [:language, :entries, :evidence]

  @type entry :: %{category: atom(), sub_type: atom(), sub_type_source: :declared | :learned}
  @type t :: %__MODULE__{language: String.t(), entries: %{String.t() => entry()}, evidence: LexicalEvidence.t()}

  @doc """
  Builds the table from the training split's `{text, intent}` rows and the
  intent registry. Every intent in `rows` must be in the registry and declare a
  category. Registry intents with no training rows are left out: the intent
  classifier is trained on these rows, so it cannot answer with them.
  """
  @spec build([{String.t(), String.t()}], map(), String.t()) :: t()
  def build([_ | _] = rows, registry, language) when is_map(registry) do
    language = Brain.Language.validate!(language)
    declared = declared!(rows, registry)

    {evidence, _report} = LexicalEvidence.select(labeled(rows, declared), language: language)

    undeclared = for {intent, {_category, nil}} <- declared, do: intent
    learned = learn_sub_types(rows, evidence, undeclared)

    entries =
      Map.new(declared, fn
        {intent, {category, nil}} ->
          {intent, %{category: category, sub_type: Map.fetch!(learned, intent), sub_type_source: :learned}}

        {intent, {category, sub_type}} ->
          {intent, %{category: category, sub_type: String.to_atom(sub_type), sub_type_source: :declared}}
      end)

    %__MODULE__{language: language, entries: entries, evidence: evidence}
  end

  @doc """
  Out-of-fold speech-act evidence for the rows `table` was built from:
  `{fold_of_row, models}`, folded as `Brain.Analysis.LexicalEvidence.fold_models/2`
  folds the same `{text, intent}` rows, so a training row's intent evidence and
  its speech-act evidence both come from models fitted without it. Fold `k`'s
  model is fitted, with the table's prior strength and labels, on the labeled
  rows of the other folds.
  """
  @spec fold_evidence([{String.t(), String.t()}], map(), t()) ::
          {[non_neg_integer()], %{non_neg_integer() => LexicalEvidence.t()}}
  def fold_evidence([_ | _] = rows, registry, %__MODULE__{evidence: evidence} = table) when is_map(registry) do
    declared = declared!(rows, registry)
    folds = LexicalEvidence.folds(rows)
    opts = [language: table.language, prior_strength: evidence.prior_strength, intents: evidence.intents]

    models =
      Map.new(0..(LexicalEvidence.fold_count() - 1), fn k ->
        train = rows |> Enum.zip(folds) |> Enum.reject(&(elem(&1, 1) == k)) |> Enum.map(&elem(&1, 0))
        {k, LexicalEvidence.fit(labeled(train, declared), opts)}
      end)

    {folds, models}
  end

  # Each training intent's `{category, declared sub-type or nil}`.
  defp declared!(rows, registry) do
    rows
    |> Enum.map(&elem(&1, 1))
    |> Enum.uniq()
    |> Map.new(fn intent ->
      case Map.fetch(registry, intent) do
        {:ok, %{"category" => category} = schema} when is_binary(category) ->
          {intent, {String.to_atom(category), schema["speech_act"]}}

        {:ok, schema} ->
          raise "SpeechActTable: registry intent #{inspect(intent)} declares no category: #{inspect(schema, limit: 5)}"

        :error ->
          raise "SpeechActTable: training intent #{inspect(intent)} is not in the intent registry"
      end
    end)
  end

  # The rows of intents that declare a sub-type, labeled "<category>:<sub_type>".
  defp labeled(rows, declared) do
    for {text, intent} <- rows, {category, sub_type} = Map.fetch!(declared, intent), sub_type != nil,
        do: {text, "#{category}:#{sub_type}"}
  end

  @doc "The `{category, sub_type}` of `intent`. Raises for an intent the table does not hold."
  @spec lookup!(t(), String.t()) :: {atom(), atom()}
  def lookup!(%__MODULE__{entries: entries, language: language}, intent) when is_binary(intent) do
    case Map.fetch(entries, intent) do
      {:ok, %{category: category, sub_type: sub_type}} -> {category, sub_type}
      :error -> raise "SpeechActTable: no #{language} speech act for intent #{inspect(intent)}"
    end
  end

  @doc "Where the table for `language` is stored."
  @spec path(String.t()) :: String.t()
  def path(language) do
    Path.join(Brain.Analysis.RunProvenance.models_path!(), "speech_act_table.#{Brain.Language.validate!(language)}.term")
  end

  @doc "Writes the table to `path/1` for its language."
  @spec save!(t()) :: :ok
  def save!(%__MODULE__{language: language} = table) do
    File.write!(path(language), Brain.ML.ModelStore.serialize(table))
  end

  @doc "Reads the stored table for `language`. Raises when it is missing or is not a table."
  @spec read!(String.t()) :: t()
  def read!(language) do
    file = path(language)

    case File.read(file) do
      {:ok, binary} ->
        case :erlang.binary_to_term(binary) do
          %__MODULE__{language: ^language, evidence: %LexicalEvidence{}} = table ->
            table

          %__MODULE__{language: ^language} ->
            raise "SpeechActTable: #{file} holds a #{language} table without its speech-act evidence, " <>
                    "written before the table kept one. Rebuild it with `mix gen_micro_data --world <id>`."

          other ->
            raise "SpeechActTable: #{file} does not hold a #{language} table: #{inspect(other, limit: 3)}"
        end

      {:error, reason} ->
        raise "SpeechActTable: cannot read #{file} (#{inspect(reason)}). Train it with `mix gen_micro_data --world <id>`."
    end
  end

  @doc "The table loaded at boot for `language`."
  @spec loaded!(String.t()) :: t()
  def loaded!(language) do
    case :persistent_term.get({__MODULE__, language}, nil) do
      %__MODULE__{} = table -> table
      nil -> raise "SpeechActTable: no #{language} table is loaded; the loader did not run or the language is unsupported"
    end
  end

  @doc false
  def put_loaded(%__MODULE__{language: language} = table), do: :persistent_term.put({__MODULE__, language}, table)

  # Each undeclared intent takes the sub-type most of its utterances are read
  # as by the speech-act evidence (fitted on the intents that declare one).
  defp learn_sub_types(rows, evidence, undeclared) do
    undeclared_set = MapSet.new(undeclared)

    rows
    |> Enum.filter(fn {_text, intent} -> MapSet.member?(undeclared_set, intent) end)
    |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))
    |> Map.new(fn {intent, texts} ->
      {majority, _count} =
        texts
        |> Enum.map(&(evidence |> top_label(&1) |> elem(1)))
        |> Enum.frequencies()
        |> Enum.max_by(fn {sub_type, count} -> {count, sub_type} end)

      {intent, majority}
    end)
  end

  @doc """
  The speech act `evidence` reads `text`'s words as most strongly: `{:ok,
  {category, sub_type}, value}`, `value` in (0, 1) being the evidence's
  strength, or `:none` when no word of the text is associated with any speech
  act (none of its n-grams is known, or none is positively associated).
  """
  @spec read(LexicalEvidence.t(), String.t()) :: {:ok, {atom(), atom()}, float()} | :none
  def read(%LexicalEvidence{} = evidence, text) when is_binary(text) do
    case evidence |> LexicalEvidence.features(text) |> Enum.zip(evidence.intents) |> Enum.max_by(&elem(&1, 0)) do
      {value, label} when value > 0 -> {:ok, split_label(label), value}
      _ -> :none
    end
  end

  defp top_label(evidence, text) do
    evidence
    |> LexicalEvidence.features(text)
    |> Enum.zip(evidence.intents)
    |> Enum.max_by(fn {value, _label} -> value end)
    |> elem(1)
    |> split_label()
  end

  defp split_label(label) do
    [category, sub_type] = String.split(label, ":", parts: 2)
    {String.to_atom(category), String.to_atom(sub_type)}
  end
end
