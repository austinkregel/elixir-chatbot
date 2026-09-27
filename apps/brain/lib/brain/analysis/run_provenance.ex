defmodule Brain.Analysis.RunProvenance do
  @moduledoc """
  Describes the environment a measurement run was produced in, so two runs can be
  compared and a difference attributed to something.

  Feature vectors and axis values are only comparable across runs that share a
  feature schema and the state the schema is derived from.

  ## What is recorded

  - `git` — the commit, and whether the tree was dirty. A dirty tree means the
    sha does not identify the code.
  - `versions` — Elixir and OTP, which affect Nx/EXLA numeric output.
  - `extractor` — `ChunkFeatures.schema_fingerprint/0` and `group_widths/0`. The
    fingerprint says whether the vector's schema moved; the widths say which
    group moved.
  - `age_graph` — a digest of `TypeHierarchy.parent_types/0`. Feature group 23's
    width is `length(parent_types()) + 2`, read from an ETS table populated from
    the AGE graph, so the vector's shape depends on graph state. Group 10 looks
    like it does too, but `Brain.Lexicon.domain_atoms/0` is a compile-time
    attribute of 45 entries and cannot move at runtime.
  - `lexicon` — the domain count and a digest. Group 10 reads it at call time
    while `EnrichmentFeatures` froze the same list at compile time, so a stale
    build is visible here.
  - `models` — SHA-256 per `.term` under the configured models path.
  - `datasets` — SHA-256 per `data/classifiers/*.json`.

  Every accessor raises rather than recording `nil`, including `git`: a record
  with a gap in it reads as comparable when it is not.
  """

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Analysis.TypeHierarchy

  @classifier_dir "classifiers"

  @doc """
  The full provenance record for the current environment.

  Raises when any component cannot be determined -- see the failure policy
  above.
  """
  @spec capture!() :: map()
  def capture! do
    %{
      captured_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      git: git!(),
      versions: versions(),
      extractor: extractor!(),
      age_graph: age_graph!(),
      lexicon: lexicon!(),
      models: models!(),
      datasets: datasets!()
    }
  end

  @doc """
  The 16-hex extractor schema fingerprint on its own.

  Callers that only need to know whether two runs share a vector schema -- the
  micro-classifier load gate, for instance -- want this rather than the whole
  record.
  """
  @spec schema_fingerprint!() :: String.t()
  def schema_fingerprint! do
    case ChunkFeatures.schema_fingerprint() do
      fp when is_binary(fp) and byte_size(fp) == 16 ->
        fp

      other ->
        raise "RunProvenance: ChunkFeatures.schema_fingerprint/0 returned #{inspect(other)}, " <>
                "expected 16 hex characters"
    end
  end

  @doc """
  SHA-256 of one file, as `%{name:, version:, sha256:}`.

  The same record shape `Brain.Training.POS.inputs/0` stamps into a POS model,
  so a model's recorded inputs and a run's recorded datasets are directly
  comparable.
  """
  @spec input!(Path.t(), String.t()) :: map()
  def input!(path, version) do
    unless File.regular?(path) do
      raise "RunProvenance: no file to hash at #{path}"
    end

    %{name: Path.basename(path), version: version, sha256: sha256_file!(path)}
  end

  @doc "SHA-256 of a file's bytes, lowercase hex. Raises when unreadable."
  @spec sha256_file!(Path.t()) :: String.t()
  def sha256_file!(path) do
    path
    |> File.stream!(65_536)
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  @doc """
  The directory holding the `.term` models, resolved exactly as
  `Brain.ML.MicroClassifiers` resolves it.
  """
  @spec models_path!() :: Path.t()
  def models_path! do
    path =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil -> Brain.priv_path("ml_models")
        configured -> configured
      end

    unless File.dir?(path) do
      raise "RunProvenance: models_path #{path} is not a directory. " <>
              "Train or download the models before taking a snapshot."
    end

    path
  end

  # -- components -------------------------------------------------------------

  defp git! do
    sha =
      case System.cmd("git", ["rev-parse", "HEAD"], stderr_to_stdout: true) do
        {out, 0} ->
          String.trim(out)

        {out, code} ->
          raise "RunProvenance: `git rev-parse HEAD` exited #{code}: #{String.trim(out)}. " <>
                  "A run whose commit cannot be identified cannot be bisected."
      end

    dirty? =
      case System.cmd("git", ["status", "--porcelain"], stderr_to_stdout: true) do
        {out, 0} -> String.trim(out) != ""
        {out, code} -> raise "RunProvenance: `git status --porcelain` exited #{code}: #{String.trim(out)}"
      end

    %{sha: sha, dirty: dirty?}
  end

  defp versions do
    %{elixir: System.version(), otp: System.otp_release()}
  end

  defp extractor! do
    widths = ChunkFeatures.group_widths()
    declared = ChunkFeatures.vector_dimension()
    summed = widths |> Enum.map(&elem(&1, 1)) |> Enum.sum()

    # The manifest is the authority for both numbers, so a disagreement means
    # group_widths/0 and vector_dimension/0 have been derived from different
    # manifests -- which would make every recorded width untrustworthy.
    unless summed == declared do
      raise "RunProvenance: group widths sum to #{summed} but vector_dimension is #{declared}"
    end

    %{
      schema_fingerprint: schema_fingerprint!(),
      vector_dimension: declared,
      group_widths: Map.new(widths, fn {group, width} -> {to_string(group), width} end)
    }
  end

  defp age_graph! do
    unless TypeHierarchy.ready?() do
      raise "RunProvenance: TypeHierarchy is not ready, so feature group 23's width is " <>
              "undefined and no vector taken now is comparable to one taken later."
    end

    case TypeHierarchy.parent_types() do
      [] ->
        raise "RunProvenance: TypeHierarchy has no parent types loaded. Feature group 23 " <>
                "would be its minimum width, which is the task 072 failure mode: the vector " <>
                "changes shape because an external store was rebuilt."

      types ->
        %{parent_type_count: length(types), parent_types_digest: digest(types)}
    end
  end

  # No emptiness guard here, unlike age_graph!/0: `domain_atoms/0` is a
  # compile-time `@domain_atoms` attribute, and the type checker rejects a `[]`
  # clause against it as unreachable. That it cannot be empty is the same fact
  # that makes group 10's width fixed; group 23 is the runtime-variable one.
  defp lexicon! do
    atoms = Brain.Lexicon.domain_atoms()
    %{domain_count: length(atoms), domains_digest: digest(atoms)}
  end

  defp models! do
    path = models_path!()

    case Path.wildcard(Path.join(path, "**/*.term")) |> Enum.sort() do
      [] ->
        raise "RunProvenance: no .term models under #{path}"

      paths ->
        Map.new(paths, fn p -> {Path.relative_to(p, path), sha256_file!(p)} end)
    end
  end

  defp datasets! do
    # Brain.data_path/1, not File.cwd!/0: an umbrella run has two working
    # directories (root for boot, apps/brain for that app's tests), so a
    # relative "data/classifiers" names two different places in one run.
    path = Brain.data_path(@classifier_dir)

    unless File.dir?(path) do
      raise "RunProvenance: no classifier training data at #{path}. " <>
              "Run `mix gen_micro_data` first, or run this from the umbrella root."
    end

    case Path.wildcard(Path.join(path, "*.json")) |> Enum.sort() do
      [] ->
        raise "RunProvenance: #{path} holds no .json training data"

      paths ->
        Map.new(paths, fn p -> {Path.basename(p), sha256_file!(p)} end)
    end
  end

  # A digest over a sorted list of atoms, so the record stays a fixed size
  # whether the list holds 15 entries or 1,500.
  #
  # Sorted for `domain_atoms/0`'s sake: it is `Map.values/1` over a
  # compile-time map, so its order is an implementation detail of the map rather
  # than a promise. `parent_types/0` already sorts, so sorting again is a no-op
  # there. An order-sensitive digest would report drift that is not there.
  defp digest(atoms) do
    atoms
    |> Enum.map(&to_string/1)
    |> Enum.sort()
    |> Enum.join("\n")
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> binary_part(0, 16)
  end
end
