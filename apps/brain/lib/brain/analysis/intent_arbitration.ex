defmodule Brain.Analysis.IntentArbitration do
  @moduledoc """
  What `Brain.Analysis.Pipeline`'s `determine_intent` does when the two trained
  classifiers disagree: `:intent_full` gives a registered intent whose
  consolidated domain (`Brain.Lexicon.IntentDomains`) is not the domain the
  `:intent_domain` classifier gives the chunk. Which of them to believe there
  is learned from data, not fixed.

  ## Routes

  A disagreement can be resolved three ways, each a *route* to an intent:

  - `:classifier` -- `:intent_full`'s intent;
  - `:domain` -- the most probable intent of `:intent_full`'s candidates that
    is registered and lies in `:intent_domain`'s domain (none when no candidate
    does: the route then gives no intent);
  - `:speech_act` -- the pipeline's speech-act route, the multi-signal scorer
    over the registry.

  ## Learning

  The examples are the training rows of the intent gold standard on which
  out-of-fold predictions of the two classifiers disagree: each row is
  analyzed by models fitted without it (`mix train_intent_arbitration`), and
  each route's intent is compared with the row's gold intent. Each example is
  `%{fold, classifier_domain, profile_domain, right: %{route => boolean}}`.

  For a group of examples -- all of them (`:global`), those of one
  `:intent_full` domain (`:classifier_domain`), of one `:intent_domain` domain
  (`:profile_domain`), or of one pair of them (`:pair`) -- the probability that
  a route is right is its rate smoothed toward the rate over every example:

      (right_g + k * right / n) / (n_g + k)

  A group no example fell in has that overall rate. The disagreement is
  resolved by the route of the highest probability in its group. When two
  routes are equally probable the data cannot decide between them, and the
  arbitration abstains: no intent is determined.

  The granularity and the prior strength `k` are chosen by cross-validation
  over the examples' folds -- each fold's disagreements resolved by counts from
  the other folds and scored by how many routes taken were right -- and the
  one-standard-error rule (`Brain.ML.OneStandardError`): of the candidates
  whose mean fold score is within one standard error of the best, the coarsest
  granularity, then the largest strength. The held-out split is never read.

  The model is fitted per language and stored beside the micro-classifiers
  (`micro/intent_arbitration.<language>.term`). `Brain.Analysis.IntentArbitration.Loader`
  loads it at boot and boot fails without it.
  """

  alias Brain.ML.OneStandardError

  @routes [:classifier, :domain, :speech_act]
  @granularities [:global, :classifier_domain, :profile_domain, :pair]
  # Pseudo-counts of disagreements, from barely more than a group's own counts
  # to a prior that outweighs every group.
  @prior_strengths [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10_000]

  @enforce_keys [:language, :routes, :granularity, :prior_strength, :groups, :overall, :fitted_examples]
  defstruct [:language, :routes, :granularity, :prior_strength, :groups, :overall, :fitted_examples, :cross_validation, :selection]

  @type route :: :classifier | :domain | :speech_act
  @type granularity :: :global | :classifier_domain | :profile_domain | :pair
  @type counts :: %{n: non_neg_integer(), right: %{route() => non_neg_integer()}}
  @type t :: %__MODULE__{
          language: String.t(),
          routes: [route()],
          granularity: granularity(),
          prior_strength: number(),
          groups: %{term() => counts()},
          overall: counts(),
          fitted_examples: non_neg_integer(),
          cross_validation: map() | nil,
          selection: map() | nil
        }

  @doc "The routes a disagreement can be resolved by, in the order ties are reported."
  @spec routes() :: [route()]
  def routes, do: @routes

  @doc """
  Fits the counts of `examples` (see the moduledoc) at `:granularity` with
  `:prior_strength`, for `:routes` (the routes every example measures) and
  `:language`. Raises on no example and on an example that does not measure
  every route.
  """
  @spec fit([map()], keyword()) :: t()
  def fit([_ | _] = examples, opts) do
    routes = routes!(Keyword.fetch!(opts, :routes))
    granularity = granularity!(Keyword.fetch!(opts, :granularity))
    strength = strength!(Keyword.fetch!(opts, :prior_strength))
    examples = Enum.map(examples, &example!(&1, routes))

    %__MODULE__{
      language: Brain.Language.validate!(Keyword.fetch!(opts, :language)),
      routes: routes,
      granularity: granularity,
      prior_strength: strength,
      groups: examples |> Enum.group_by(&group(granularity, &1.classifier_domain, &1.profile_domain)) |> Map.new(fn {g, es} -> {g, counts(es, routes)} end),
      overall: counts(examples, routes),
      fitted_examples: length(examples)
    }
  end

  def fit([], _opts), do: raise(ArgumentError, "IntentArbitration: no disagreements to fit")

  @doc """
  Chooses the granularity and the prior strength (see the moduledoc) on
  `examples`, each carrying its `:fold`, and fits on all of them. Returns
  `{model, report}`; the report, kept as the model's `cross_validation`, gives
  every candidate's fold scores, mean and standard error, and the rule's
  choice is the model's `selection`. Fewer than two folds holding an example
  raise. Options: `:routes` and `:language`, as for `fit/2`.
  """
  @spec select([map()], keyword()) :: {t(), map()}
  def select([_ | _] = examples, opts) do
    routes = routes!(Keyword.fetch!(opts, :routes))
    examples = Enum.map(examples, &example!(&1, routes))
    folds = examples |> Enum.map(& &1.fold) |> Enum.uniq() |> Enum.sort()

    if length(folds) < 2 do
      raise ArgumentError, "IntentArbitration: only #{length(folds)} fold(s) hold a disagreement; the one-standard-error rule needs two"
    end

    candidates = for granularity <- @granularities, strength <- @prior_strengths, do: {granularity, strength}

    scored =
      Enum.map(candidates, fn {granularity, strength} = candidate ->
        scores =
          Enum.map(folds, fn k ->
            {held, training} = Enum.split_with(examples, &(&1.fold == k))
            model = fit(training, routes: routes, granularity: granularity, prior_strength: strength, language: Keyword.fetch!(opts, :language))
            Enum.count(held, &right?(model, &1)) / length(held)
          end)

        {candidate, scores}
      end)

    rule = OneStandardError.choose(scored, fn {granularity, strength} -> {coarseness(granularity), strength} end)
    {granularity, strength} = rule.chosen

    report = %{
      folds: folds,
      examples: length(examples),
      candidates: Map.new(scored, fn {candidate, scores} -> {candidate, Map.put(Map.fetch!(rule.scores, candidate), :folds, scores)} end)
    }

    model =
      examples
      |> fit(routes: routes, granularity: granularity, prior_strength: strength, language: Keyword.fetch!(opts, :language))
      |> Map.merge(%{cross_validation: report, selection: Map.drop(rule, [:scores])})

    {model, report}
  end

  def select([], _opts), do: raise(ArgumentError, "IntentArbitration: no disagreements to select from")

  @doc """
  The route a disagreement between `classifier_domain` (`:intent_full`'s) and
  `profile_domain` (`:intent_domain`'s) is resolved by: `{:route, route,
  probabilities}`, or `{:abstain, probabilities}` when the most probable routes
  are equally probable.
  """
  @spec decide(t(), String.t(), String.t()) :: {:route, route(), %{route() => float()}} | {:abstain, %{route() => float()}}
  def decide(%__MODULE__{} = model, classifier_domain, profile_domain) when is_binary(classifier_domain) and is_binary(profile_domain) do
    probabilities = probabilities(model, classifier_domain, profile_domain)
    best = probabilities |> Map.values() |> Enum.max()

    case Enum.filter(model.routes, &(Map.fetch!(probabilities, &1) == best)) do
      [route] -> {:route, route, probabilities}
      _tied -> {:abstain, probabilities}
    end
  end

  @doc "Each route's smoothed probability of being right in the disagreement's group."
  @spec probabilities(t(), String.t(), String.t()) :: %{route() => float()}
  def probabilities(%__MODULE__{} = model, classifier_domain, profile_domain) do
    %{n: n_g, right: right_g} = Map.get(model.groups, group(model.granularity, classifier_domain, profile_domain), empty(model.routes))
    k = model.prior_strength

    Map.new(model.routes, fn route ->
      overall = Map.fetch!(model.overall.right, route) / model.overall.n
      {route, (Map.fetch!(right_g, route) + k * overall) / (n_g + k)}
    end)
  end

  @doc """
  Where the model for `language` is stored: beside the micro-classifiers, in
  `micro/`. Like them it consumes the feature vector's classifications rather
  than feeding the vector, so it is not among the upstream models a vector's
  provenance hashes (`Brain.Analysis.RunProvenance.vector_environment!/0`).
  """
  @spec path(String.t()) :: String.t()
  def path(language) do
    Path.join([Brain.Analysis.RunProvenance.models_path!(), "micro", "intent_arbitration.#{Brain.Language.validate!(language)}.term"])
  end

  @doc "Writes `model` to `path/1` for its language."
  @spec save!(t()) :: :ok
  def save!(%__MODULE__{language: language} = model), do: File.write!(path(language), Brain.ML.ModelStore.serialize(model))

  @doc "Reads the stored model for `language`. Raises when it is missing or is not a model."
  @spec read!(String.t()) :: t()
  def read!(language) do
    file = path(language)

    case File.read(file) do
      {:ok, binary} ->
        case :erlang.binary_to_term(binary) do
          %__MODULE__{language: ^language} = model -> model
          other -> raise "IntentArbitration: #{file} does not hold a #{language} model: #{inspect(other, limit: 3)}"
        end

      {:error, reason} ->
        raise "IntentArbitration: cannot read #{file} (#{inspect(reason)}). Train it with `mix train_intent_arbitration --world <id>`."
    end
  end

  @doc "The model `Brain.Analysis.IntentArbitration.Loader` loaded at boot for `language`."
  @spec loaded!(String.t()) :: t()
  def loaded!(language) do
    case :persistent_term.get({__MODULE__, language}, nil) do
      %__MODULE__{} = model -> model
      nil -> raise "IntentArbitration: no #{language} model is loaded; the loader did not run or the language is unsupported"
    end
  end

  @doc false
  def put_loaded(%__MODULE__{language: language} = model), do: :persistent_term.put({__MODULE__, language}, model)

  # Whether the route `model` takes for `example` is right; an abstention is not.
  defp right?(model, example) do
    case decide(model, example.classifier_domain, example.profile_domain) do
      {:route, route, _p} -> Map.fetch!(example.right, route)
      {:abstain, _p} -> false
    end
  end

  defp group(:global, _c, _p), do: :global
  defp group(:classifier_domain, c, _p), do: {:classifier_domain, c}
  defp group(:profile_domain, _c, p), do: {:profile_domain, p}
  defp group(:pair, c, p), do: {:pair, c, p}

  # Larger is coarser, and so more regularized.
  defp coarseness(:global), do: 2
  defp coarseness(:classifier_domain), do: 1
  defp coarseness(:profile_domain), do: 1
  defp coarseness(:pair), do: 0

  defp counts(examples, routes) do
    %{n: length(examples), right: Map.new(routes, fn route -> {route, Enum.count(examples, &Map.fetch!(&1.right, route))} end)}
  end

  defp empty(routes), do: %{n: 0, right: Map.new(routes, &{&1, 0})}

  defp example!(%{classifier_domain: c, profile_domain: p, right: right} = example, routes)
       when is_binary(c) and is_binary(p) and is_map(right) do
    case Enum.reject(routes, &is_boolean(Map.get(right, &1))) do
      [] -> example
      missing -> raise ArgumentError, "IntentArbitration: an example does not say whether #{inspect(missing)} was right: #{inspect(example, limit: 6)}"
    end
  end

  defp example!(other, _routes) do
    raise ArgumentError, "IntentArbitration: an example needs :classifier_domain, :profile_domain and :right, got #{inspect(other, limit: 6)}"
  end

  defp routes!([_ | _] = routes) do
    case routes -- @routes do
      [] -> Enum.filter(@routes, &(&1 in routes))
      unknown -> raise ArgumentError, "IntentArbitration: unknown routes #{inspect(unknown)}; the routes are #{inspect(@routes)}"
    end
  end

  defp routes!(other), do: raise(ArgumentError, "IntentArbitration: :routes must be a list of routes, got #{inspect(other)}")

  defp granularity!(g) when g in @granularities, do: g
  defp granularity!(other), do: raise(ArgumentError, "IntentArbitration: unknown granularity #{inspect(other)}")

  defp strength!(k) when is_number(k) and k > 0, do: k
  defp strength!(other), do: raise(ArgumentError, "IntentArbitration: :prior_strength must be a positive number, got #{inspect(other)}")
end
