defmodule Brain.Test.Singletons do
  @moduledoc """
  Shared process state a test replaces, snapshotted before and put back after.

  The Sandbox rolls back the database; it does nothing about what the brain
  keeps in processes. Several of those are one instance for the whole VM, so a
  test that rebuilds one changes what every later test sees — and which tests
  those are depends on the run's order, which is what makes the damage hard to
  read afterwards.

  Call these from `setup` in the test that does the replacing.
  """

  alias Brain.Analysis.LearningStore
  alias Brain.Epistemic.BeliefStore
  alias Brain.Epistemic.JTMS
  alias Brain.Epistemic.SourceAuthority
  alias Brain.Epistemic.UserModelStore
  alias Brain.Memory.Embedder
  alias Brain.Test.HTTPSnapshot

  @doc """
  Snapshots `keys` of the named ETS `table` and puts them back when the test
  ends — whatever the test inserts, overwrites or deletes under those keys.

  For tests that write into a table the application owns, such as the
  gazetteer's `:gazetteer_entities` and `:gazetteer_prefixes`. Deleting the
  keys afterwards is not a restore: most words a test "adds" are already in
  the gazetteer's sources, and `Gazetteer.remove_entry/1` deletes the whole
  entry, so the run continues without the real data.
  """
  @spec preserve_ets_keys!(atom(), [term()]) :: :ok
  def preserve_ets_keys!(table, keys) when is_atom(table) and is_list(keys) do
    snapshot = Enum.map(keys, fn key -> {key, :ets.lookup(table, key)} end)

    ExUnit.Callbacks.on_exit(fn ->
      if :ets.whereis(table) != :undefined do
        Enum.each(snapshot, fn {key, rows} ->
          :ets.delete(table, key)
          Enum.each(rows, &:ets.insert(table, &1))
        end)
      end
    end)

    :ok
  end

  @doc """
  Snapshots the global embedder's vocabulary and IDF weights, and restores
  them when the test ends.

  For a test that calls `Embedder.build_vocabulary/1`: the vocabulary is
  global, and a small one built from a handful of texts leaves every later
  text embedding as a zero vector, which reads downstream as "no similarity"
  rather than as an error (`Corroborator.compare_claims/2` returns 0.0,
  `ChunkSegmenter` falls back to a tie-break).
  """
  @spec preserve_embedder!() :: :ok
  def preserve_embedder! do
    {:ok, snapshot} = Embedder.export_model()

    ExUnit.Callbacks.on_exit(fn ->
      # The embedder is a named singleton; it may have been restarted since.
      if Process.whereis(Embedder), do: :ok = Embedder.load_model(snapshot)
    end)

    :ok
  end

  @doc """
  Loads the named HTTP snapshots for this test and puts the snapshot server
  back the way it was when the test ends.

  `HTTPSnapshot.use_snapshot/1` inserts into one VM-wide named ETS table, and
  nothing ever removed an entry, so the loaded set only grew across a run.
  `MockHTTP` asks the server for a response (`mock_http.ex:47`) and the server
  answers from the first `:ets.tab2list/1` entry that matches, under a
  deliberately loose matcher — substring on the path, and *any one* overlapping
  param value is enough (`matches_request?/3`). So a snapshot left behind by an
  earlier test can answer a later test's request with a stale recorded body.

  The fixtures make that concrete: `open_alex/search_transformer` and
  `open_alex/search_cs` are both `https://api.openalex.org/works` and both
  carry `per_page: 3`, so either will match the other's request, and which one
  wins is `:ets.tab2list/1` order on a `:set`.

  Fails loudly if a named snapshot is missing, rather than leaving the test to
  hit the matcher's "no snapshot" path.
  """
  @spec use_http_snapshots!([String.t()]) :: :ok
  def use_http_snapshots!(names) when is_list(names) do
    snapshot = HTTPSnapshot.export_snapshots()

    ExUnit.Callbacks.on_exit(fn ->
      if Process.whereis(HTTPSnapshot), do: :ok = HTTPSnapshot.load_snapshots(snapshot)
    end)

    Enum.each(names, fn name ->
      {:ok, _} = HTTPSnapshot.use_snapshot(name)
    end)

    :ok
  end

  @doc """
  Snapshots every learned-analysis parameter and puts the values back when the
  test ends.

  `LearningStore` is a single GenServer holding one map for the whole VM, and
  its consumers read the live values: `SemanticChunker` reads `"chunker"`
  (`semantic_chunker.ex:77`) and `ResponseGate` reads
  `"response_optionality"` (`analysis/response_gate.ex:305`). Without this,
  `learning_store_test.exs` left `max_chunk_words` at 60, `bot_names` at
  `["test"]` and `speech_acts.confidence_threshold` at 0.35 for the rest of the
  run.

  It also writes that map to `config :brain, :learning_params_path`, which is
  now the absolute `apps/brain/test/data/learned_params.json` (`config/test.exs`),
  so the file the store loads at `init/1` is the file it saves to. It used to
  be the *relative* `test/data/learned_params.json`, resolved against the
  current working directory, and there are two of those in a run: Mix boots the
  applications from the umbrella root and then runs this app's tests from
  `apps/brain`, so the store read the root copy and wrote the `apps/brain` one.
  Restoring the file is what keeps a test's values off disk.

  Restore is `update_params/3` with `admin: true` per component, which is the
  only whole-value write the module exposes; it bypasses `admin_locked`, and
  because the snapshot map carries `"admin_locked"` itself the lock state is
  restored with everything else. Two limits, both named rather than hidden:

  * `update_params/3` always stamps `"learned_at"` with the current time
    (`learning_store.ex:165`), so that one field comes back as a timestamp
    rather than its snapshot value. Nothing outside `learning_store_test.exs`
    reads it — the only other `learned_at` in the tree belongs to
    `Brain.FactDatabase.Fact`, an unrelated struct.
  * A *new* top-level component a test invents cannot be removed: the module
    has no delete. No test does this today; all of them write components that
    exist in `default_params/0`.

  The file is rewritten from the snapshot too, so the residue on disk matches
  the restored state even when the run ends before the 5 s save debounce
  (`@save_debounce_ms`, `learning_store.ex:22`) fires.
  """
  @spec preserve_learning_params!() :: :ok
  def preserve_learning_params! do
    snapshot = LearningStore.get_all_params()
    path = Application.get_env(:brain, :learning_params_path)

    ExUnit.Callbacks.on_exit(fn ->
      if Process.whereis(LearningStore) do
        snapshot
        |> Enum.filter(fn {_component, value} -> is_map(value) end)
        |> Enum.each(fn {component, value} ->
          :ok = LearningStore.update_params(component, value, admin: true)
        end)
      end

      if is_binary(path) do
        path |> Path.dirname() |> File.mkdir_p!()
        File.write!(path, Jason.encode!(snapshot, pretty: true))
      end
    end)

    :ok
  end

  @doc """
  Clears the named epistemic stores now and again when the test ends, so a test
  that needs an empty store neither inherits nor leaves anything.

  `stores` is any of `:beliefs`, `:jtms`, `:user_models`, `:source_authority`.

  Clearing on the way out *is* the restore here, because an empty store is the
  boot state — PROVEN, not assumed:

  * `BeliefStore.init/1` fills itself from `load_from_atlas/1`
    (`belief_store.ex:278, 635-658`) and `UserModelStore.init/1` does the same
    (`user_model.ex:112-119`); the prepared test database holds **0 rows** in
    both `atlas_test.atlas_beliefs` and `atlas_test.atlas_user_models`, so both
    boot empty.
  * `JTMS.init/1` builds its state literally, with no load of any kind
    (`jtms.ex:140-153`), so it boots empty by construction.
  * `SourceAuthority.clear/0` wipes `tracking` only
    (`source_authority.ex:206-209`); the bootstrap `profiles` map is untouched,
    so the tier definitions every consumer reads survive a clear. `tracking` is
    the learned credibility counts, loaded at `init/1` by `load_learned_data/1`
    from `atlas_test.atlas_source_authority` (`source_authority.ex:360-394`),
    which holds **0 rows**, so it too boots empty.

  Everything these stores hold mid-run was therefore put there by an earlier
  test. That is the leak: beliefs live in GenServer state, which the Sandbox
  does not roll back, so they survived into every later test until some other
  test's `clear/0` happened to remove them. For `SourceAuthority` the leak is
  the same shape with a different payload: `record_outcome/2` bumps a tier's
  confirmed/contradicted/added counts, and `effective_confidence/1` multiplies
  the tier's `initial_confidence` by the credibility those counts produce
  (`source_authority.ex:142-151`), so a test that ran 20 `:contradicted`
  outcomes against `:mentor` leaves every later reader a lower confidence than
  the tier declares.
  """
  @spec reset_epistemic_stores!([:beliefs | :jtms | :user_models | :source_authority]) :: :ok
  def reset_epistemic_stores!(stores) when is_list(stores) do
    clear = fn ->
      Enum.each(stores, fn
        :beliefs -> if Process.whereis(BeliefStore), do: BeliefStore.clear()
        :jtms -> if Process.whereis(JTMS), do: JTMS.clear()
        :user_models -> if Process.whereis(UserModelStore), do: UserModelStore.clear_all()
        :source_authority -> if Process.whereis(SourceAuthority), do: SourceAuthority.clear()
      end)
    end

    clear.()
    ExUnit.Callbacks.on_exit(clear)

    :ok
  end
end
