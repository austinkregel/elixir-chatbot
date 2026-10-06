defmodule Brain.ML.MicroProvenance do
  @moduledoc """
  Records what a micro-classifier was trained on, and refuses a model whose
  training data or feature schema has changed since.

  The record lives inside the `.term` model rather than in a side-car file, so it
  cannot be separated from the model it describes. `Brain.Training.POS` does the
  same for the POS tagger. `Brain.ML.ModelStore.serialize/2` encodes
  deterministically, so a content hash identifies a model.

  ## What is recorded

      %{
        inputs: [%{name: "intent_full.json", version: "repo:data/classifiers", sha256: "..."}],
        schema_fingerprint: "bc289842ba5ccb3d",   # feature-vector classifiers only
        environment: %{digest: "...", components: %{...}},  # likewise
        git_sha: "341b3d3...",
        at: "2026-09-25T19:07:46.356907Z"
      }

  `schema_fingerprint` and `environment` are recorded only for
  `kind: :feature_vector` models. A text classifier consumes strings, so renaming a
  feature dimension cannot affect it, and recording either would make it fail for
  no reason. Whether a model is feature-vector-backed is read from its own `:kind`
  field rather than from a list of names.

  ## Three questions, three checks

  The training data, the vector's schema and the vector's values are independent,
  and a model can be stale in any one of them:

  | check | question |
  |---|---|
  | `check_inputs!/3` | was the training file changed after training? |
  | `check_schema!/3` | do the 337 dimensions still mean the same things? |
  | `check_environment!/3` | does the same text still produce the same numbers? |

  The third exists because the first two can both pass while the model is useless.
  Measured 2026-09-28: all six deployed feature-vector models had matching input
  hashes and matching schema fingerprints, and **none** of their stored training
  vectors reproduced — `mem_novelty` stored 0.999 against 0.5 recomputed,
  `sa_assertive` 1.0 against 0.0. See `RunProvenance.vector_environment!/0` for
  what the vector's values depend on.

  ## Where the checks run

  `check_inputs!/3` reads only the filesystem and can run anywhere.
  `check_schema!/3` needs `Brain.Analysis.TypeHierarchy` to be ready, because
  feature group 23 takes its dimension names from the AGE graph — so the feature
  vector's schema is not knowable at boot. `check_environment!/3` needs the
  lexicon, the gazetteer and the graph, so it is later still.

  `Brain.ML.MicroClassifiers` runs only `check_inputs!/3` at load, and records a
  failure as a value rather than raising: a gate that raises while models are being
  loaded cannot be bootstrapped, because training runs the pipeline.
  `Brain.ML.ModelPreflight` runs `check_current!/3` once the supervision tree is
  up, which is the earliest point all three can be answered.
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
      record
      |> Map.put(:schema_fingerprint, RunProvenance.schema_fingerprint!())
      |> Map.put(:environment, training_environment!(name))
    else
      record
    end
  end

  @doc """
  Raises unless `model` was trained on the data as it is now.

  Returns `:ok`; there is no error-tuple form, so a caller cannot choose to
  ignore a mismatch. Reads only the filesystem.
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

      The model's training data changed after it was trained. Retrain with
      `mix train_micro`.
      """
    end

    :ok
  end

  @doc """
  Raises unless `model` was trained under the extractor schema in force now.

  Only meaningful for `kind: :feature_vector` models; a text classifier passes
  unconditionally.

  Requires `Brain.Analysis.TypeHierarchy` to be ready.
  `ChunkFeatures.schema_fingerprint/0` walks `dimension_manifest/0`, whose group
  23 resolves its names from `TypeHierarchy.parent_types/0` — an ETS table
  populated from the AGE graph — and raises when that table is empty. Call this
  no earlier than the first classification.
  """
  @spec check_schema!(map(), atom() | String.t(), Path.t()) :: :ok
  def check_schema!(model, name, path) when is_map(model) do
    if feature_vector?(model), do: do_check_schema!(model, name, path), else: :ok
  end

  @doc """
  Raises unless the state the vector's *values* depend on is what it was at
  training time.

  `check_schema!/3` asks whether the dimensions still mean the same things.
  This asks the separate question of whether the same text still produces the same
  numbers, which the schema check cannot see: measured 2026-09-28, none of the six
  deployed models' stored training vectors reproduced while every schema
  fingerprint matched.

  Only meaningful for `kind: :feature_vector` models. Requires the app to be
  running, because it reads the lexicon, the gazetteer and the AGE graph.
  """
  @spec check_environment!(map(), atom() | String.t(), Path.t()) :: :ok
  def check_environment!(model, name, path) when is_map(model) do
    if feature_vector?(model), do: do_check_environment!(model, name, path), else: :ok
  end

  @doc """
  All three checks, for callers that are past boot and can reach the stores.
  """
  @spec check_current!(map(), atom() | String.t(), Path.t()) :: :ok
  def check_current!(model, name, path) when is_map(model) do
    :ok = check_inputs!(model, name, path)
    :ok = check_schema!(model, name, path)
    check_environment!(model, name, path)
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

  @doc "The path to the environment record `mix gen_micro_data` writes beside the data."
  @spec training_environment_path(atom() | String.t()) :: Path.t()
  def training_environment_path(name) do
    Brain.data_path(Path.join(@classifier_dir, "#{name}.environment.json"))
  end

  @doc """
  The environment the training vectors in `name`'s data were computed under.

  Read from the sidecar `mix gen_micro_data` writes, not from the live stores: the
  record has to describe the state the *vectors* were built in, which is generation
  time, not training time. Stamping the live environment at training time would
  assert something about data it did not compute.

  The data file's own hash is what keeps the two together — regenerate the data
  without its sidecar and `check_inputs!/3` fires on the changed hash.
  """
  @spec training_environment!(atom() | String.t()) :: map()
  def training_environment!(name) do
    path = training_environment_path(name)

    unless File.regular?(path) do
      raise """
      no environment record for micro-classifier #{name} at #{path}.

      The feature vectors in #{Path.basename(training_data_path(name))} were computed
      against some state of the POS tagger, the speech-act voter, the lexicon, the
      gazetteer and the AGE graph, and without that record there is no way to tell
      whether this runtime still reproduces them.

      Run `mix gen_micro_data` to regenerate the data and its environment together.
      """
    end

    case path |> File.read!() |> Jason.decode!() do
      %{"digest" => digest} = record when is_binary(digest) ->
        %{digest: digest, components: Map.get(record, "components", %{})}

      other ->
        raise "MicroProvenance: #{path} is not an environment record: " <>
                "#{inspect(other) |> String.slice(0, 160)}"
    end
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
        fingerprint, so there is no way to tell whether its 337 dimensions still
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

  defp do_check_environment!(model, name, path) do
    recorded = get_in(model, [:training, :environment])
    current = RunProvenance.vector_environment!()

    cond do
      is_nil(recorded) ->
        raise """
        feature-vector classifier #{name} at #{path} records nothing about the state
        its training vectors were computed against, so there is no way to tell
        whether the same text still produces the same numbers.

        The vector is built from a Pipeline.analyze_chunk/2 result, so it embeds the
        output of the POS tagger, the speech-act voter, the sentiment and entity
        models, and reads the lexicon, the gazetteer and the AGE graph. None of
        that is covered by the schema fingerprint.

        The environment now digests to #{current.digest}. Retrain with
        `mix train_micro`.
        """

      recorded.digest != current.digest ->
        raise """
        feature-vector classifier #{name} at #{path} was trained against
        environment #{recorded.digest}, but the environment now digests to
        #{current.digest}.

        The vector's schema is unchanged, so the dimensions still mean what they
        meant -- but the same text no longer produces the same values, which is
        what the model's centroids were fitted to.

        #{format_environment_drift(recorded.components, current.components)}
        Retrain with `mix train_micro`.
        """

      true ->
        :ok
    end
  end

  # Names the components that moved, so the failure points at a cause rather than
  # just asserting a digest mismatch.
  #
  # Both sides are rendered with string keys first: the recorded half came back
  # through JSON and the current half is freshly built with atom keys, so comparing
  # them as-is would report every component as changed.
  defp format_environment_drift(recorded, current) do
    recorded = stringify(recorded)
    current = stringify(current)
    keys = (Map.keys(recorded) ++ Map.keys(current)) |> Enum.uniq() |> Enum.sort()

    keys
    |> Enum.map(fn key ->
      was = Map.get(recorded, key)
      now = Map.get(current, key)

      if was == now do
        "  #{key}: unchanged"
      else
        "  #{key}:\n      trained: #{inspect(was)}\n      now:     #{inspect(now)}"
      end
    end)
    |> Enum.join("\n")
    |> then(&(&1 <> "\n"))
  end

  defp stringify(%{} = map) do
    Map.new(map, fn {k, v} -> {to_string(k), stringify(v)} end)
  end

  defp stringify(other), do: other

  defp git_sha do
    case System.cmd("git", ["rev-parse", "HEAD"], stderr_to_stdout: true) do
      {out, 0} -> String.trim(out)
      {out, code} -> raise "MicroProvenance: `git rev-parse HEAD` exited #{code}: #{String.trim(out)}"
    end
  end
end
