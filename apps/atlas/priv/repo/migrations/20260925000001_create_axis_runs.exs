defmodule Atlas.Repo.Migrations.CreateAxisRuns do
  use Ecto.Migration

  def change do
    # One row per `mix axes.snapshot`. `provenance` holds the whole
    # Brain.Analysis.RunProvenance record: commit and dirty flag, the
    # extractor's schema fingerprint and per-group widths, the AGE graph's
    # parent-type digest, the lexicon digest, and SHA-256 per model and
    # dataset. It is stored as one jsonb blob rather than columns because the
    # set of things worth recording grows, and a schema migration per addition
    # would discourage recording.
    #
    # The fields that *are* columns are the ones runs get selected by.
    create table(:atlas_axis_runs, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :provenance, :map, null: false, default: %{}
      add :schema_fingerprint, :string, null: false
      add :corpus_sha256, :string, null: false
      add :utterance_count, :integer, null: false
      add :tag, :string
      add :tag_note, :text

      timestamps(type: :utc_datetime_usec)
    end

    # Tagging promotes a run to a durable reference point, so a tag names
    # exactly one run. Untagged runs accumulate freely and are prunable, which
    # is why the index is partial rather than a plain unique index over a
    # nullable column.
    create unique_index(:atlas_axis_runs, [:tag], where: "tag IS NOT NULL", name: :atlas_axis_runs_tag_index)

    # Comparison is always run-vs-run or run-vs-tag, and two runs are only
    # comparable when their extractor schema matches -- so this is the index
    # that answers "what can I compare this against?"
    create index(:atlas_axis_runs, [:schema_fingerprint])
    create index(:atlas_axis_runs, [:inserted_at])

    # One row per utterance per axis: 18 axes x ~5,274 utterances is ~95k rows
    # for a full run. `value` is text for every axis, including the continuous
    # ones, because an axis's kind is declared in ChunkProfile.axis_manifest/0
    # and duplicating it as a column type here would be a second source of
    # truth that can disagree with the manifest.
    create table(:atlas_axis_observations, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :run_id, references(:atlas_axis_runs, type: :binary_id, on_delete: :delete_all), null: false
      add :utterance_id, :string, null: false
      add :axis, :string, null: false
      add :value, :string, null: false
      add :status, :string, null: false
      add :reason, :string

      # The full 343-float vector, recorded only for tagged runs. Without it a
      # regression like task 072 -- where three memory dimensions inverted
      # while the vector length stayed 343 -- is visible as an axis changing
      # but not attributable to a feature.
      add :feature_vector, {:array, :float}

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_axis_observations, [:run_id, :axis])
    create index(:atlas_axis_observations, [:run_id, :utterance_id])

    # The per-axis censuses this exists to serve all group by status, and the
    # interesting query is "which axes were never computed on this run".
    create index(:atlas_axis_observations, [:run_id, :axis, :status])
  end
end
