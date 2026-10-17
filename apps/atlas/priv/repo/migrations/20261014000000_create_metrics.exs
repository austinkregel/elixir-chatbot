defmodule Atlas.Repo.Migrations.CreateMetrics do
  use Ecto.Migration

  # One store for every metric that must survive a restart, in measurement
  # windows, replacing the semantic-fact metric tables it generalizes.
  #
  # A window belongs to a family of metrics (the semantic facts, the node's
  # telemetry) in one scope: a world, or the whole node when `world_id` is
  # NULL. Its values are added to while it is open (no `ended_at`); starting a
  # new window ends the open one and keeps its values, so one window can be
  # compared with the next. At most one open window per family and scope.
  #
  # The semantic-fact windows and counts are copied over, ids kept, into the
  # "semantic_facts" family, and the old tables dropped. No column defaults:
  # every write names its family, scope, start, name and value.
  def up do
    create table(:atlas_metric_windows, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string
      add :family, :string, null: false
      add :started_at, :utc_datetime_usec, null: false
      add :ended_at, :utc_datetime_usec
    end

    create index(:atlas_metric_windows, [:family, :world_id, :started_at])
    create index(:atlas_metric_windows, [:world_id])

    # NULL scopes the node; COALESCE makes the node's open window as unique as
    # a world's. A world is never named "" (checked below), so it cannot clash.
    create unique_index(:atlas_metric_windows, ["family", "COALESCE(world_id, '')"],
             where: "ended_at IS NULL",
             name: :atlas_metric_windows_open_index
           )

    create constraint(:atlas_metric_windows, :world_must_not_be_blank,
             check: "world_id IS NULL OR btrim(world_id) <> ''"
           )

    create constraint(:atlas_metric_windows, :family_must_not_be_blank, check: "btrim(family) <> ''")

    # One named value of one window, combined with each new value by its
    # aggregate: added ("sum"), or the smaller ("min") or larger ("max") kept.
    # A "sum" row exists only once something was counted, so it is positive.
    create table(:atlas_metric_values, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string

      add :window_id, references(:atlas_metric_windows, type: :binary_id, on_delete: :delete_all),
        null: false

      add :name, :string, null: false
      add :aggregate, :string, null: false
      add :value, :bigint, null: false

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_metric_values, [:window_id, :name])
    create index(:atlas_metric_values, [:world_id])

    create constraint(:atlas_metric_values, :aggregate_must_be_known,
             check: "aggregate IN ('sum', 'min', 'max')"
           )

    create constraint(:atlas_metric_values, :sum_must_be_positive,
             check: "aggregate <> 'sum' OR value > 0"
           )

    # The latest record of one name in one window (the last training run of a
    # model, say), replaced by each newer one.
    create table(:atlas_metric_records, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string

      add :window_id, references(:atlas_metric_windows, type: :binary_id, on_delete: :delete_all),
        null: false

      add :name, :string, null: false
      add :record, :map, null: false
      add :recorded_at, :utc_datetime_usec, null: false
    end

    create unique_index(:atlas_metric_records, [:window_id, :name])
    create index(:atlas_metric_records, [:world_id])

    execute("""
    INSERT INTO atlas_metric_windows (id, world_id, family, started_at, ended_at)
    SELECT id, world_id, 'semantic_facts', started_at, ended_at
    FROM atlas_semantic_fact_metric_windows
    """)

    execute("""
    INSERT INTO atlas_metric_values (id, world_id, window_id, name, aggregate, value, inserted_at, updated_at)
    SELECT id, world_id, window_id, counter, 'sum', count, inserted_at, updated_at
    FROM atlas_semantic_fact_metric_counts
    """)

    drop table(:atlas_semantic_fact_metric_counts)
    drop table(:atlas_semantic_fact_metric_windows)
  end

  def down do
    create table(:atlas_semantic_fact_metric_windows, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false
      add :started_at, :utc_datetime_usec, null: false
      add :ended_at, :utc_datetime_usec
    end

    create index(:atlas_semantic_fact_metric_windows, [:world_id, :started_at])

    create unique_index(:atlas_semantic_fact_metric_windows, [:world_id],
             where: "ended_at IS NULL",
             name: :atlas_semantic_fact_metric_windows_open_index
           )

    create table(:atlas_semantic_fact_metric_counts, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false

      add :window_id,
          references(:atlas_semantic_fact_metric_windows, type: :binary_id, on_delete: :delete_all),
          null: false

      add :counter, :string, null: false
      add :count, :bigint, null: false

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_semantic_fact_metric_counts, [:window_id, :counter])
    create index(:atlas_semantic_fact_metric_counts, [:world_id])

    create constraint(:atlas_semantic_fact_metric_counts, :count_must_be_positive, check: "count > 0")

    # Only the semantic-fact family had a home before; the other families'
    # windows, values and records are dropped with the new tables.
    execute("""
    INSERT INTO atlas_semantic_fact_metric_windows (id, world_id, started_at, ended_at)
    SELECT id, world_id, started_at, ended_at
    FROM atlas_metric_windows
    WHERE family = 'semantic_facts'
    """)

    execute("""
    INSERT INTO atlas_semantic_fact_metric_counts (id, world_id, window_id, counter, count, inserted_at, updated_at)
    SELECT v.id, v.world_id, v.window_id, v.name, v.value, v.inserted_at, v.updated_at
    FROM atlas_metric_values v
    JOIN atlas_metric_windows w ON w.id = v.window_id
    WHERE w.family = 'semantic_facts'
    """)

    drop table(:atlas_metric_records)
    drop table(:atlas_metric_values)
    drop table(:atlas_metric_windows)
  end
end
