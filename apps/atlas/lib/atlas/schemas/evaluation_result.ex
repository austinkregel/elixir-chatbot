defmodule Atlas.Schemas.EvaluationResult do
  @moduledoc """
  Ecto schema for evaluation results.

  Replaces file-based evaluation result storage in priv/evaluation/results/.
  Stores accuracy, F1 scores, and per-class metrics for each evaluation task run.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_evaluation_results" do
    field :task, :string
    field :accuracy, :float
    field :macro_f1, :float
    field :weighted_f1, :float
    field :total_examples, :integer
    field :duration_ms, :integer
    field :per_class, :map, default: %{}
    field :confusion_matrix, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(task accuracy macro_f1 total_examples)a
  @optional_fields ~w(weighted_f1 duration_ms per_class confusion_matrix)a

  def changeset(result, attrs) do
    result
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:task, ["intent", "ner", "sentiment", "speech_act"])
  end
end
