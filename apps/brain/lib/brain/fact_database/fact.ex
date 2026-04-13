defmodule Brain.FactDatabase.Fact do
  @moduledoc """
  Struct representing a verifiable fact in the fact database.

  This struct enforces a consistent schema for facts, with:
  - `entity` - The subject/topic of the fact (e.g., "France", "water")
  - `entity_type` - Classification of the entity (e.g., "country", "substance")

  Use `from_map/1` to convert legacy map formats (which may use inconsistent
  field names) into properly typed structs.
  """

  @type t :: %__MODULE__{
          id: String.t(),
          entity: String.t(),
          entity_type: String.t() | nil,
          fact: String.t(),
          category: String.t(),
          confidence: float(),
          verification_source: String.t() | nil,
          learned_at: integer() | nil
        }

  @enforce_keys [:id, :entity, :fact, :category]
  defstruct [
    :id,
    :entity,
    :entity_type,
    :fact,
    :category,
    :verification_source,
    :learned_at,
    confidence: 1.0
  ]

  @doc """
  Create a Fact struct from a map, normalizing field names.

  Handles backwards compatibility with maps that use:
  - `"entity_type"` as the subject name (legacy format)
  - `"entity"` as the subject name (correct format)

  The `entity_type` classification field is inferred from the category
  and entity name when not explicitly provided.
  """
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    # Handle both "entity" and "entity_type" as subject name for backwards compat
    # Priority: "entity" (correct) > "entity_type" (legacy) > "unknown"
    entity = get_entity_name(map)
    category = map["category"] || "unknown"

    %__MODULE__{
      id: map["id"] || generate_id(),
      entity: entity,
      entity_type: infer_entity_type(entity, category),
      fact: map["fact"] || "",
      category: category,
      confidence: map["confidence"] || 1.0,
      verification_source: map["verification_source"],
      learned_at: map["learned_at"]
    }
  end

  @doc """
  Create a new Fact with the given attributes.
  """
  @spec new(keyword()) :: t()
  def new(attrs) when is_list(attrs) do
    entity = Keyword.fetch!(attrs, :entity)
    category = Keyword.get(attrs, :category, "learned")

    %__MODULE__{
      id: Keyword.get_lazy(attrs, :id, &generate_id/0),
      entity: entity,
      entity_type: Keyword.get(attrs, :entity_type) || infer_entity_type(entity, category),
      fact: Keyword.fetch!(attrs, :fact),
      category: category,
      confidence: Keyword.get(attrs, :confidence, 1.0),
      verification_source: Keyword.get(attrs, :verification_source),
      learned_at: Keyword.get(attrs, :learned_at)
    }
  end

  @doc """
  Convert a Fact struct to a map for JSON serialization.

  Nil values are excluded from the output.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = fact) do
    %{
      "id" => fact.id,
      "entity" => fact.entity,
      "entity_type" => fact.entity_type,
      "fact" => fact.fact,
      "category" => fact.category,
      "confidence" => fact.confidence,
      "verification_source" => fact.verification_source,
      "learned_at" => fact.learned_at
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  # Private functions

  defp get_entity_name(map) do
    cond do
      # Prefer explicit "entity" field (correct format)
      is_binary(map["entity"]) and map["entity"] != "" ->
        map["entity"]

      # Fall back to "entity_type" if it looks like a subject name (legacy format)
      is_binary(map["entity_type"]) and map["entity_type"] != "" ->
        map["entity_type"]

      true ->
        "unknown"
    end
  end

  defp generate_id do
    "fact_#{:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)}"
  end

  @doc """
  Infer the entity type (classification) from the entity name and category.

  Uses category-based heuristics and common patterns to determine the type.
  """
  @spec infer_entity_type(String.t(), String.t()) :: String.t()
  def infer_entity_type(entity, category) do
    # Combine entity name and category as input for the classifier
    input = "#{String.downcase(entity)} #{String.downcase(category)}"

    case Brain.ML.MicroClassifiers.classify(:entity_type, input) do
      {:ok, label, _score} -> label
      _ -> String.downcase(category)
    end
  end
end
