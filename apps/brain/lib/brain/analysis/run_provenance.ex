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
  The state a feature vector's *values* depend on, as `%{digest, components}`.

  `schema_fingerprint!/0` says whether the vector's 337 dimensions still mean the
  same things. It says nothing about whether the same text still produces the same
  numbers, and that is a separate question with a separate answer: measured
  2026-09-28, the deployed models' stored training vectors do not reproduce, while
  the vector is otherwise deterministic — identical within a process and across a
  fresh one. The schema had not moved. What moved was this.

  The vector is computed from a `Pipeline.analyze_chunk/2` result, so it inherits
  everything that analysis reads:

  - `upstream_models` — the `.term` models whose *output* the vector embeds. The
    POS tagger feeds `pos_distribution`, the supersense groups and, through the
    speech-act voter, `speech_act` and `speech_act_wh_interaction`; the sentiment
    and entity models feed their own groups. Only the top level of the models
    directory is read: `micro/` holds the classifiers that *consume* vectors, and
    hashing those into their own provenance would be circular, while `default/`,
    `lattice/`, `lstm/` and `ouro/` are response-side and never reach a vector.
  - `lexicon` — WordNet's loaded size. `lexical_domains` is 45 dimensions and the
    three supersense groups another 45. `hyp_cache_size` is deliberately excluded:
    it is a cache that grows as queries run, so including it would report drift on
    every process that had done some work.
  - `gazetteer` — entity and prefix counts, which feed the `entity` group.
  - `age_graph` — the same parent-type digest `age_graph!/0` records, because
    group 23's names come from it.

  `load_time_ms` is excluded from both stats maps for the same reason as the
  hypernym cache: it is a property of the run, not of the data.

  The digest is for comparison; the components are kept so a mismatch can say
  which one moved, exactly as `extractor!/0` keeps `group_widths` beside its
  fingerprint.
  """
  @spec vector_environment!() :: map()
  def vector_environment! do
    components = %{
      upstream_models: upstream_models!(),
      lexicon: lexicon_state!(),
      gazetteer: gazetteer_state!(),
      age_graph: age_graph!()
    }

    %{digest: map_digest(components), components: components}
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

  # Top level only, and not recursive: see vector_environment!/0 on why micro/ and
  # the response-side subdirectories are excluded.
  defp upstream_models! do
    path = models_path!()

    case path |> Path.join("*.term") |> Path.wildcard() |> Enum.sort() do
      [] ->
        raise "RunProvenance: no upstream .term models directly under #{path}. The feature " <>
                "vector embeds their output, so a vector taken now cannot be compared to one " <>
                "taken when they were present."

      paths ->
        Map.new(paths, fn p -> {Path.basename(p), sha256_file!(p)} end)
    end
  end

  defp lexicon_state! do
    stats = Brain.ML.Lexicon.stats()

    for key <- [:word_count, :synset_count, :hypernym_count, :morph_count], into: %{} do
      case Map.get(stats, key) do
        n when is_integer(n) and n > 0 ->
          {key, n}

        other ->
          raise "RunProvenance: Brain.ML.Lexicon.stats/0 reports #{key}=#{inspect(other)}. " <>
                  "A vector whose lexical groups were built against an unloaded lexicon is not " <>
                  "comparable to one built against a loaded one."
      end
    end
  end

  defp gazetteer_state! do
    stats = Brain.ML.Gazetteer.stats()

    unless Map.get(stats, :loaded) == true do
      raise "RunProvenance: the gazetteer reports loaded=#{inspect(Map.get(stats, :loaded))}, " <>
              "so the entity feature group would be built against nothing."
    end

    # `entities` is excluded: it counts the loaded gazetteer plus whatever
    # discovery has synced from the graph since boot, so it varies by run. The
    # gazetteer file is hashed in upstream_models/0 instead. Entities the graph
    # contributes after boot are therefore not covered here.
    for key <- [:prefixes, :entity_types], into: %{} do
      case Map.get(stats, key) do
        n when is_integer(n) and n > 0 -> {key, n}
        other -> raise "RunProvenance: Brain.ML.Gazetteer.stats/0 reports #{key}=#{inspect(other)}"
      end
    end
  end

  # Digests a nested map of scalars by rendering it in sorted key order, so the
  # result depends on the contents and not on map iteration order.
  defp map_digest(map) do
    map
    |> render_sorted()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> binary_part(0, 16)
  end

  defp render_sorted(map) when is_map(map) do
    map
    |> Enum.sort_by(fn {k, _v} -> to_string(k) end)
    |> Enum.map_join("\n", fn {k, v} -> "#{k}=#{render_sorted(v)}" end)
  end

  defp render_sorted(list) when is_list(list), do: Enum.map_join(list, ",", &render_sorted/1)
  defp render_sorted(other), do: to_string(other)

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
