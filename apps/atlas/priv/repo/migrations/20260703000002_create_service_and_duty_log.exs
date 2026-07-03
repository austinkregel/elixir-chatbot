defmodule Atlas.Repo.Migrations.CreateServiceAndDutyLog do
  use Ecto.Migration

  def change do
    # ── append-only milestone service record ────────────────────────────────
    create table(:atlas_service_records, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :soul_id, :string, null: false
      add :agent_id, :string
      add :kind, :string, null: false
      add :order_id, :string
      add :world_id, :string
      add :under_order_of, :string
      add :outcome, :string
      add :reason, :string
      add :authored_by, :string, null: false, default: "runtime"
      add :payload, :map, default: %{}
      add :occurred_at, :utc_datetime_usec
      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_service_records, [:soul_id])
    create index(:atlas_service_records, [:kind])
    create index(:atlas_service_records, [:order_id])
    create index(:atlas_service_records, [:occurred_at])

    # ── per-soul rehydration snapshot + counters (upsert target) ────────────
    create table(:atlas_service_summaries) do
      add :soul_id, :string, null: false
      add :duty_status, :string, null: false, default: "active"
      add :rank, :string, default: "ensign"
      add :home_world_id, :string
      add :mind_world_id, :string
      add :co_id, :string
      add :reports, {:array, :string}, default: []
      add :standing_grants, {:array, :string}, default: []
      add :current_assignment, :map, default: %{}
      add :served_since, :utc_datetime_usec
      add :last_commissioned_at, :utc_datetime_usec
      add :orders_completed, :integer, default: 0
      add :orders_dissented, :integer, default: 0
      add :orders_failed, :integer, default: 0
      add :reliefs, :integer, default: 0
      add :achievements, {:array, :map}, default: []
      add :metadata, :map, default: %{}
      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_service_summaries, [:soul_id])

    # ── agent-authored duty log ─────────────────────────────────────────────
    create table(:atlas_duty_log_entries, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :soul_id, :string, null: false
      add :order_id, :string
      add :world_id, :string
      add :note, :string, null: false
      add :tags, {:array, :string}, default: []
      add :authored_by, :string, null: false, default: "agent"
      add :payload, :map, default: %{}
      add :occurred_at, :utc_datetime_usec
      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_duty_log_entries, [:soul_id])
    create index(:atlas_duty_log_entries, [:order_id])
    create index(:atlas_duty_log_entries, [:occurred_at])
  end
end
