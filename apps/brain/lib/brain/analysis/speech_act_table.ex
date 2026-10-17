defmodule Brain.Analysis.SpeechActTable do
  @moduledoc """
  The speech act of each intent: its category and sub-type, used when the
  intent classifier's answer refines a chunk's speech act.

  The intent registry (`priv/analysis/intent_registry.json`) declares a
  `"category"` for every intent and a `"speech_act"` sub-type for most. An
  intent whose sub-type is not declared gets one learned from the data: a
  `Brain.Analysis.LexicalEvidence` model is fitted on the training utterances of
  the intents that do declare one, labeled with their sub-type, and each
  undeclared intent takes the sub-type most of its own training utterances are
  read as. Each entry records whether its sub-type was declared or learned.

  This replaces a list of intent-name prefixes in the speech-act classifier,
  which named domains ("greeting", "farewell") that no intent in the registry
  uses, so every greeting intent read as an assertive statement.

  Built per language at training time (`speech_act_table.<language>.term`,
  directly under the models directory) and loaded at boot.
  """

  alias Brain.Analysis.LexicalEvidence

  @enforce_keys [:language, :entries]
  defstruct [:language, :entries]

  @type entry :: %{category: atom(), sub_type: atom(), sub_type_source: :declared | :learned}
  @type t :: %__MODULE__{language: String.t(), entries: %{String.t() => entry()}}

  @doc """
  Builds the table from the training split's `{text, intent}` rows and the
  intent registry. Every intent in `rows` must be in the registry and declare a
  category. Registry intents with no training rows are left out: the intent
  classifier is trained on these rows, so it cannot answer with them.
  """
  @spec build([{String.t(), String.t()}], map(), String.t()) :: t()
  def build([_ | _] = rows, registry, language) when is_map(registry) do
    language = Brain.Language.validate!(language)
    intents = rows |> Enum.map(&elem(&1, 1)) |> Enum.uniq()

    declared =
      Map.new(intents, fn intent ->
        case Map.fetch(registry, intent) do
          {:ok, %{"category" => category} = schema} when is_binary(category) ->
            {intent, {String.to_atom(category), schema["speech_act"]}}

          {:ok, schema} ->
            raise "SpeechActTable: registry intent #{inspect(intent)} declares no category: #{inspect(schema, limit: 5)}"

          :error ->
            raise "SpeechActTable: training intent #{inspect(intent)} is not in the intent registry"
        end
      end)

    undeclared = for {intent, {_category, nil}} <- declared, do: intent

    learned =
      case undeclared do
        [] -> %{}
        _ -> learn_sub_types(rows, declared, undeclared, language)
      end

    entries =
      Map.new(declared, fn
        {intent, {category, nil}} ->
          {intent, %{category: category, sub_type: Map.fetch!(learned, intent), sub_type_source: :learned}}

        {intent, {category, sub_type}} ->
          {intent, %{category: category, sub_type: String.to_atom(sub_type), sub_type_source: :declared}}
      end)

    %__MODULE__{language: language, entries: entries}
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
          %__MODULE__{language: ^language} = table -> table
          other -> raise "SpeechActTable: #{file} does not hold a #{language} table: #{inspect(other, limit: 3)}"
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

  # A sub-type classifier fitted on the utterances of intents that declare
  # one; each undeclared intent takes the sub-type most of its utterances get.
  defp learn_sub_types(rows, declared, undeclared, language) do
    labeled =
      for {text, intent} <- rows, {_category, sub_type} = declared[intent], sub_type != nil, do: {text, sub_type}

    {model, _accuracies} = LexicalEvidence.select(labeled, language: language)

    undeclared_set = MapSet.new(undeclared)

    rows
    |> Enum.filter(fn {_text, intent} -> MapSet.member?(undeclared_set, intent) end)
    |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))
    |> Map.new(fn {intent, texts} ->
      {majority, _count} =
        texts
        |> Enum.map(&top_label(model, &1))
        |> Enum.frequencies()
        |> Enum.max_by(fn {label, count} -> {count, label} end)

      {intent, String.to_atom(majority)}
    end)
  end

  defp top_label(model, text) do
    model
    |> LexicalEvidence.features(text)
    |> Enum.zip(model.intents)
    |> Enum.max_by(fn {value, _label} -> value end)
    |> elem(1)
  end
end
