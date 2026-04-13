defmodule Atlas.Repo.Migrations.CreateEvaluationResults do
  use Ecto.Migration

  def change do
    create table(:atlas_evaluation_results, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :task, :string, null: false
      add :accuracy, :float, null: false
      add :macro_f1, :float, null: false
      add :weighted_f1, :float
      add :total_examples, :integer, null: false
      add :duration_ms, :integer
      add :per_class, :map, default: %{}
      add :confusion_matrix, :map, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_evaluation_results, [:task])
    create index(:atlas_evaluation_results, [:task, :inserted_at])
  end
end
