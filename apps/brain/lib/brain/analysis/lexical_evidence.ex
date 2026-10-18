defmodule Brain.Analysis.LexicalEvidence do
  @moduledoc """
  What the words of an utterance say about its intent, learned from the intent
  gold standard's training split.

  The feature vector's other groups describe structure -- parts of speech,
  speech act, entities, lexicon domains -- and say little about which words
  were used. So a greeting the speech-act classifier had no rule for ("howdy",
  "top of the morning") looked like any other short exclamation. This group
  carries the words: for every intent, how strongly the utterance's 1- and
  2-grams are associated with it in the training data.

  ## The association

  For n-gram `w` and intent `g`, the log-odds of `w` in `g`'s utterances against
  every other intent's, with an informative Dirichlet prior (Monroe, Colaresi and
  Quinn, 2008): the prior gives each n-gram `prior_strength × its share of all
  n-grams`, so a rare n-gram is pulled toward no association rather than taken
  at face value. The difference is divided by its standard error, giving a
  z-score.

  An utterance's value for intent `g` is the mean z-score of its known n-grams
  for `g`, squashed into (-1, 1) by `z / (1 + |z|)`. An utterance with no known
  n-gram scores 0 for every intent.

  ## Choosing the prior strength

  `select/2` compares candidate prior strengths by k-fold cross-validation on
  the training rows: each fold that holds a row is scored by how often the
  intent with the highest evidence is the row's intent. The strength is
  chosen by the one-standard-error rule (`Brain.ML.OneStandardError`): of the
  strengths whose mean fold accuracy is within one standard error of the best
  mean, the largest. The accuracy is flat across the largest strengths, and
  the highest mean among them is chosen by fold noise. Fewer than two folds
  holding a row raise. The held-out split is never read.

  ## Language

  A model is fitted for one language and stored per language
  (`lexical_evidence.<language>.term`, directly under the models directory, so
  run provenance hashes it). Tokens come from `Brain.ML.Tokenizer`, which is the
  English implementation, so English is the only language that can be fitted.
  """

  alias Brain.ML.{OneStandardError, Tokenizer}

  @enforce_keys [:language, :prior_strength, :intents, :scores]
  defstruct [:language, :prior_strength, :intents, :scores, :fitted_rows]

  @type t :: %__MODULE__{
          language: String.t(),
          prior_strength: number(),
          intents: [String.t()],
          scores: %{String.t() => [float()]},
          fitted_rows: non_neg_integer()
        }

  # Spans weak to effectively unbounded smoothing: on the intent training split
  # (3,655 rows, 2026-10-08) cross-validated accuracy rose from 56% at 1 to 64%
  # and was flat from 100,000 on, so the grid ends past that plateau. The
  # one-standard-error rule chooses among them.
  @prior_strengths [1, 10, 100, 1000, 3000, 10_000, 30_000, 100_000, 300_000]
  @folds 5

  @doc """
  Fits a model from `{text, intent}` rows.

  Options: `:language` (required), `:prior_strength` (required), and `:intents`,
  the intents the model scores, in order. It defaults to the intents of `rows`;
  a fold model passes the full model's, so an intent whose rows all fell in the
  held-out fold keeps its place in the vector. Every intent of `rows` must be
  in it.
  """
  @spec fit([{String.t(), String.t()}], keyword()) :: t()
  def fit([_ | _] = rows, opts) do
    language = Brain.Language.validate!(Keyword.fetch!(opts, :language))
    prior_strength = Keyword.fetch!(opts, :prior_strength)
    row_intents = rows |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> Enum.sort()
    intents = Keyword.get(opts, :intents, row_intents)

    case row_intents -- intents do
      [] -> :ok
      unknown -> raise ArgumentError, "LexicalEvidence: rows have intents outside :intents: #{inspect(unknown)}"
    end

    by_intent = Enum.group_by(rows, &elem(&1, 1), &ngrams(elem(&1, 0)))

    counts_by_intent =
      Map.new(intents, fn intent -> {intent, by_intent |> Map.get(intent, []) |> List.flatten() |> Enum.frequencies()} end)
    totals = Enum.reduce(counts_by_intent, %{}, fn {_i, c}, acc -> Map.merge(acc, c, fn _k, a, b -> a + b end) end)
    n_total = totals |> Map.values() |> Enum.sum()
    n_by_intent = Map.new(counts_by_intent, fn {i, c} -> {i, c |> Map.values() |> Enum.sum()} end)

    scores =
      Map.new(totals, fn {gram, y_w} ->
        a_w = prior_strength * y_w / n_total

        z_scores =
          Enum.map(intents, fn intent ->
            y_gw = Map.get(counts_by_intent[intent], gram, 0)
            n_g = n_by_intent[intent]
            y_rw = y_w - y_gw
            n_r = n_total - n_g

            delta =
              :math.log((y_gw + a_w) / (n_g + prior_strength - y_gw - a_w)) -
                :math.log((y_rw + a_w) / (n_r + prior_strength - y_rw - a_w))

            delta / :math.sqrt(1 / (y_gw + a_w) + 1 / (y_rw + a_w))
          end)

        {gram, z_scores}
      end)

    %__MODULE__{
      language: language,
      prior_strength: prior_strength,
      intents: intents,
      scores: scores,
      fitted_rows: length(rows)
    }
  end

  def fit([], _opts), do: raise(ArgumentError, "LexicalEvidence: no rows to fit")

  @doc """
  Picks the prior strength by #{@folds}-fold cross-validation on `rows` and the
  one-standard-error rule (see the moduledoc), then fits on all of them.
  Returns `{model, report}`: `accuracy` maps each prior strength to its
  accuracy over every row, `folds` to its accuracy on each fold that holds a
  row, and `one_standard_error` is the rule's choice -- the chosen and best
  strengths, the best's mean and standard error, the threshold, the folds
  scored and every strength's mean and standard error.
  """
  @spec select([{String.t(), String.t()}], keyword()) :: {t(), map()}
  def select(rows, opts) do
    language = Keyword.fetch!(opts, :language)
    folds = fold_assignment(rows)

    splits = for k <- 0..(@folds - 1), {held, train} <- [split_fold(rows, folds, k)], held != [], do: {k, held, train}

    if length(splits) < 2 do
      raise ArgumentError,
            "LexicalEvidence: only #{length(splits)} of #{@folds} folds hold a row; " <>
              "the one-standard-error rule needs at least two scored folds"
    end

    # Each strength's correct count per scored fold.
    correct =
      Map.new(@prior_strengths, fn strength ->
        {strength,
         Enum.map(splits, fn {_k, held, train} ->
           model = fit(train, language: language, prior_strength: strength)
           Enum.count(held, fn {text, intent} -> top_intent(model, text) == intent end)
         end)}
      end)

    by_fold =
      Map.new(correct, fn {strength, counts} ->
        {strength, counts |> Enum.zip(splits) |> Enum.map(fn {c, {_k, held, _train}} -> c / length(held) end)}
      end)

    # A larger prior strength is more regularized.
    rule =
      @prior_strengths
      |> Enum.map(&{&1, Map.fetch!(by_fold, &1)})
      |> OneStandardError.choose(& &1)

    report = %{
      accuracy: Map.new(correct, fn {strength, counts} -> {strength, Enum.sum(counts) / length(rows)} end),
      folds: by_fold,
      one_standard_error: Map.put(rule, :scored_folds, Enum.map(splits, &elem(&1, 0)))
    }

    {fit(rows, language: language, prior_strength: rule.chosen), report}
  end

  @doc """
  Out-of-fold models for the rows `model` was fitted on: `{fold_of_row,
  models}`, where a row's fold model was fitted without that row, with
  `model`'s language, prior strength and intents. A training row's features come
  from its fold model, so they never reflect its own label.
  """
  @spec fold_models([{String.t(), String.t()}], t()) :: {[non_neg_integer()], %{non_neg_integer() => t()}}
  def fold_models(rows, %__MODULE__{} = model) do
    if length(rows) != model.fitted_rows do
      raise ArgumentError,
            "LexicalEvidence: #{length(rows)} rows given for a model fitted on #{model.fitted_rows}; fold models must use the same rows"
    end

    folds = fold_assignment(rows)
    opts = [language: model.language, prior_strength: model.prior_strength, intents: model.intents]

    models =
      Map.new(0..(@folds - 1), fn k ->
        {_held, train} = split_fold(rows, folds, k)
        {k, fit(train, opts)}
      end)

    {folds, models}
  end

  @doc "The evidence vector for `text`: one value in (-1, 1) per intent, in `model.intents` order."
  @spec features(t(), String.t()) :: [float()]
  def features(%__MODULE__{intents: intents, scores: scores}, text) when is_binary(text) do
    known = text |> ngrams() |> Enum.flat_map(fn gram -> List.wrap(Map.get(scores, gram)) |> wrap_row() end)

    case known do
      [] ->
        List.duplicate(0.0, length(intents))

      rows ->
        n = length(rows)

        rows
        |> Enum.zip_with(&Enum.sum/1)
        |> Enum.map(fn sum -> squash(sum / n) end)
    end
  end

  @doc "The feature names, one per intent."
  @spec names(t()) :: [atom()]
  def names(%__MODULE__{intents: intents}), do: Enum.map(intents, &String.to_atom("lex_" <> &1))

  @doc "Where the model for `language` is stored."
  @spec path(String.t()) :: String.t()
  def path(language) do
    Path.join(Brain.Analysis.RunProvenance.models_path!(), "lexical_evidence.#{Brain.Language.validate!(language)}.term")
  end

  @doc "Writes `model` to `path/1` for its language."
  @spec save!(t()) :: :ok
  def save!(%__MODULE__{language: language} = model) do
    File.write!(path(language), Brain.ML.ModelStore.serialize(model))
  end

  @doc "Reads the stored model for `language`. Raises when it is missing or is not a model."
  @spec read!(String.t()) :: t()
  def read!(language) do
    file = path(language)

    case File.read(file) do
      {:ok, binary} ->
        case :erlang.binary_to_term(binary) do
          %__MODULE__{language: ^language} = model -> model
          other -> raise "LexicalEvidence: #{file} does not hold a #{language} model: #{inspect(other, limit: 3)}"
        end

      {:error, reason} ->
        raise "LexicalEvidence: cannot read #{file} (#{inspect(reason)}). Train it with `mix gen_micro_data --world <id>`."
    end
  end

  @doc "The model `Brain.Analysis.LexicalEvidence.Loader` loaded at boot for `language`."
  @spec loaded!(String.t()) :: t()
  def loaded!(language) do
    case :persistent_term.get({__MODULE__, language}, nil) do
      %__MODULE__{} = model -> model
      nil -> raise "LexicalEvidence: no #{language} model is loaded; the loader did not run or the language is unsupported"
    end
  end

  @doc false
  def put_loaded(%__MODULE__{language: language} = model), do: :persistent_term.put({__MODULE__, language}, model)

  # Word, number and clitic tokens, lowercased, with their adjacent pairs.
  defp ngrams(text) do
    tokens = Tokenizer.tokenize_normalized(text)
    bigrams = tokens |> Enum.chunk_every(2, 1, :discard) |> Enum.map(&Enum.join(&1, " "))
    tokens ++ bigrams
  end

  defp wrap_row([]), do: []
  defp wrap_row(row), do: [row]

  defp squash(z), do: z / (1 + abs(z))

  defp top_intent(model, text) do
    model
    |> features(text)
    |> Enum.zip(model.intents)
    |> Enum.max_by(fn {value, _intent} -> value end)
    |> elem(1)
  end

  @doc """
  The fold of each row, in order, in `0..fold_count() - 1`. Deterministic: a
  row's fold depends only on the row, so a model fitted alongside this one on
  the same rows (`Brain.Analysis.SpeechActTable.fold_evidence/3`) folds them
  the same way.
  """
  @spec folds([{String.t(), String.t()}]) :: [non_neg_integer()]
  def folds(rows), do: fold_assignment(rows)

  @doc "How many folds the rows are split into."
  @spec fold_count() :: pos_integer()
  def fold_count, do: @folds

  defp fold_assignment(rows), do: Enum.map(rows, fn row -> :erlang.phash2(row, @folds) end)

  defp split_fold(rows, folds, k) do
    rows
    |> Enum.zip(folds)
    |> Enum.split_with(fn {_row, fold} -> fold == k end)
    |> then(fn {held, train} -> {Enum.map(held, &elem(&1, 0)), Enum.map(train, &elem(&1, 0))} end)
  end
end
