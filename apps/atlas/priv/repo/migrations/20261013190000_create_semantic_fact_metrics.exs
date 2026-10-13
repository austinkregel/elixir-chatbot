defmodule Atlas.Repo.Migrations.CreateSemanticFactMetrics do
  use Ecto.Migration

  def change do
    # The measurement windows of each world's semantic-fact metrics. A world's
    # counts go to its open window (no `ended_at`); starting a new window ends
    # the open one and keeps its counts, so the responses of one window can be
    # compared with the next (before and after a retrain, say). At most one
    # open window per world. No column defaults: a window starts when it is
    # written, and the world and start are named on every write.
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

    # One counter of one window, such as the turns that retrieved semantic
    # facts or the retrieved facts in one similarity bucket. A row exists only
    # once something was counted, so `count` is never zero. No column defaults:
    # the world, window, counter and count are named on every write.
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
  end
end
