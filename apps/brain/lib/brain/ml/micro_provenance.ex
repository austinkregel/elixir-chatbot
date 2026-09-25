defmodule Brain.ML.MicroProvenance do
  @moduledoc """
  Stamps a micro-classifier with what it was trained on, and refuses to load one
  whose training data has moved since.

  This is task 072's gate. Six feature-vector classifiers — `intent_full`,
  `intent_domain`, `tense_class`, `aspect_class`, `urgency`, `certainty_level` —
  were trained on vectors whose memory dimensions later inverted, because those
  dimensions derive from the AGE knowledge graph and the graph was rebuilt. The
  model learned "this entity is familiar" and is now fed "this entity is
  maximally novel" for the same input. Nothing noticed, for three reasons:

    * the vector *length* was unchanged at 343, so no dimension-mismatch error
      fired;
    * `apps/brain/priv/ml_models/manifest.json` recorded per-file SHA-256 but
      **nothing ever read it** — one writer, zero readers;
    * `mix train_micro` never wrote that manifest at all, so it drifted further
      every time models were retrained without a full `mix train`.

  ## Why the record lives inside the model

  `Brain.Training.POS` already settled this for the tagger: provenance is
  stamped into the model term at `Brain.ML.POSTagger.save_model/2` and checked at
  load by `Brain.Training.POS.check_current!/2`. A side-car manifest has to be
  kept in sync with two directories by whoever remembers; a model that carries
  its own provenance cannot be separated from it.

  `Brain.ML.ModelStore.serialize/2` already encodes with `:deterministic`, and
  its own docs say why — "comparing file hashes compares models". That
  precondition holds on every write path in the repo, so content hashing is
  sound here without further work.

  ## What is recorded

      %{
        inputs: [%{name: "intent_full.json", version: "repo:data/classifiers", sha256: "..."}],
        schema_fingerprint: "bc289842ba5ccb3d",   # feature-vector classifiers only
        git_sha: "341b3d3...",
        at: "2026-09-25T19:07:46.356907Z"
      }

  `schema_fingerprint` is recorded **only** for `kind: :feature_vector` models.
  A text classifier consumes strings, not the feature vector, so a fingerprint
  change cannot affect it — recording one would manufacture a false mismatch
  every time an unrelated feature dimension was renamed, and the first thing
  anyone would do about a false mismatch is stop trusting the gate.

  Whether a model is feature-vector-backed is read from its own `:kind` field
  rather than from a list of classifier names, so the two cannot disagree.
  """

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Analysis.RunProvenance

  @classifier_dir "classifiers"

  @doc """
  Returns `model` with a `:training` provenance record attached.

  Raises when the training data for `name` is not on disk: a model stamped with
  provenance for data that does not exist would pass its own gate and mean
  nothing.
  """
  @spec stamp!(map(), atom() | String.t()) :: map()
  def stamp!(model, name) when is_map(model) do
    Map.put(model, :training, build!(model, name))
  end

  @doc "The provenance record for `name`, without attaching it."
  @spec build!(map(), atom() | String.t()) :: map()
  def build!(model, name) when is_map(model) do
    record = %{
      inputs: [training_input!(name)],
      git_sha: git_sha(),
      at: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    if feature_vector?(model) do
      Map.put(record, :schema_fingerprint, RunProvenance.schema_fingerprint!())
    else
      record
    end
  end

  @doc """
  Raises unless `model` was trained on the data as it is now.

  Returns `:ok`. There is deliberately no error-tuple form: a caller that can
  pattern-match the failure can also choose to ignore it, which is how
  `manifest.json` ended up with zero readers.

  This is the half of the gate that can run at **load** time. It touches only
  the filesystem, so it has no dependency on any other process being up.
  """
  @spec check_inputs!(map(), atom() | String.t(), Path.t()) :: :ok
  def check_inputs!(model, name, path) when is_map(model) do
    input = training_input!(name)
    recorded = recorded_inputs(model)
    current = %{input.name => input.sha256}

    unless recorded == current do
      raise """
      micro-classifier #{name} at #{path} is stale.

        trained on : #{inspect(recorded)}
        on disk now: #{inspect(current)}

      This is task 072's failure mode: the model's training data changed under
      it, and until now nothing checked. Retrain with `mix train_micro`.
      """
    end

    :ok
  end

  @doc """
  Raises unless `model` was trained under the extractor schema in force now.

  Only meaningful for `kind: :feature_vector` models; a text classifier passes
  unconditionally.

  **This cannot run at load time**, and the reason is worth stating because it
  is a finding rather than an inconvenience.
  `ChunkFeatures.schema_fingerprint/0` walks `dimension_manifest/0`, whose group
  23 resolves its names from `Brain.Analysis.TypeHierarchy.parent_types/0` —
  an ETS table populated from the AGE graph. `MicroClassifiers` starts earlier
  in the supervision tree than `TypeHierarchy`, so at load time the fingerprint
  is not merely unknown, it raises.

  In other words the feature vector's schema is **not knowable at boot**. That
  is the same runtime dependency that makes the width drift silently, seen from
  the other side. So this check runs at the first classification instead, which
  is both the earliest point it is computable — producing a feature vector
  requires `TypeHierarchy` to be ready — and the first point at which a wrong
  answer could actually be returned.
  """
  @spec check_schema!(map(), atom() | String.t(), Path.t()) :: :ok
  def check_schema!(model, name, path) when is_map(model) do
    if feature_vector?(model), do: do_check_schema!(model, name, path), else: :ok
  end

  @doc """
  Both halves, for callers that are past boot and can compute the schema.
  """
  @spec check_current!(map(), atom() | String.t(), Path.t()) :: :ok
  def check_current!(model, name, path) when is_map(model) do
    :ok = check_inputs!(model, name, path)
    check_schema!(model, name, path)
  end

  @doc """
  Whether a model carries a provenance record at all.

  Models trained before this existed do not. Callers use this to tell "stale"
  from "never stamped", which need different messages: the first is fixed by
  retraining, the second means the model predates the gate.
  """
  @spec stamped?(map()) :: boolean()
  def stamped?(model) when is_map(model) do
    case Map.get(model, :training) do
      %{inputs: [_ | _]} -> true
      _ -> false
    end
  end

  @doc "The path to a classifier's training data."
  @spec training_data_path(atom() | String.t()) :: Path.t()
  def training_data_path(name) do
    Brain.data_path(Path.join(@classifier_dir, "#{name}.json"))
  end

  @doc """
  The `%{name, version, sha256}` record for a classifier's training data.

  Same shape as `Brain.Training.POS.inputs/0`, so a POS model's recorded inputs
  and a micro model's are directly comparable.
  """
  @spec training_input!(atom() | String.t()) :: map()
  def training_input!(name) do
    path = training_data_path(name)

    unless File.regular?(path) do
      raise """
      no training data for micro-classifier #{name} at #{path}.

      Run `mix gen_micro_data` to produce it. A model cannot be stamped with,
      or checked against, data that is not there.
      """
    end

    %{
      name: Path.basename(path),
      version: "repo:data/#{@classifier_dir}",
      sha256: RunProvenance.sha256_file!(path)
    }
  end

  # -- internals --------------------------------------------------------------

  defp feature_vector?(model), do: Map.get(model, :kind) == :feature_vector

  defp recorded_inputs(model) do
    model
    |> Map.get(:training, %{})
    |> Map.get(:inputs, [])
    |> List.wrap()
    |> Map.new(fn input -> {input.name, input.sha256} end)
  end

  defp do_check_schema!(model, name, path) do
    recorded = get_in(model, [:training, :schema_fingerprint])
    current = ChunkFeatures.schema_fingerprint()

    cond do
      is_nil(recorded) ->
        raise """
        feature-vector classifier #{name} at #{path} records no extractor schema
        fingerprint, so there is no way to tell whether its 343 dimensions still
        mean what they meant at training time.

        The current schema is #{current}. Retrain with `mix train_micro`.
        """

      recorded != current ->
        raise """
        feature-vector classifier #{name} at #{path} was trained under extractor
        schema #{recorded}, but the extractor now emits #{current}.

        The vector length is unchanged, so nothing else would have caught this --
        the dimensions were renamed or reordered underneath the model. Note that
        feature group 23 takes its dimension names from the AGE graph at runtime,
        so this can move without any code change.

        Compare ChunkFeatures.group_widths/0 against the recorded widths to find
        which group moved, then retrain with `mix train_micro`.
        """

      true ->
        :ok
    end
  end

  defp git_sha do
    case System.cmd("git", ["rev-parse", "HEAD"], stderr_to_stdout: true) do
      {out, 0} -> String.trim(out)
      {out, code} -> raise "MicroProvenance: `git rev-parse HEAD` exited #{code}: #{String.trim(out)}"
    end
  end
end
