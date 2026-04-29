defmodule Brain.ML.TrainingData.Schemas do
  @moduledoc """
  Per-record-kind validators for training data edits.

  Each validator returns `:ok` or `{:error, reason}` so the catalog can
  reject malformed writes before they hit disk.
  """

  @doc "Validate a record for the given record kind."
  @spec validate(atom(), map()) :: :ok | {:error, String.t()}
  def validate(:intent_example, rec) when is_map(rec) do
    text = Map.get(rec, "text")
    label = Map.get(rec, "intent") || Map.get(rec, "speech_act") || Map.get(rec, "sentiment")

    cond do
      !is_binary(text) or String.trim(text) == "" ->
        {:error, "Record must have a non-empty \"text\" field"}

      !is_binary(label) or String.trim(label) == "" ->
        {:error, "Record must have a non-empty label field (intent, speech_act, or sentiment)"}

      true ->
        :ok
    end
  end

  def validate(:text_classifier_row, rec) when is_map(rec) do
    text = Map.get(rec, "text")
    label = Map.get(rec, "label")

    cond do
      !is_binary(text) or String.trim(text) == "" ->
        {:error, "Record must have a non-empty \"text\" field"}

      !is_binary(label) or String.trim(label) == "" ->
        {:error, "Record must have a non-empty \"label\" field"}

      true ->
        :ok
    end
  end

  def validate(:fv_classifier_row, _rec) do
    {:error, "Feature-vector records are build artifacts and cannot be edited directly"}
  end

  def validate(:registry_entry, {key, val}) when is_binary(key) and is_map(val) do
    domain = Map.get(val, "domain")

    cond do
      String.trim(key) == "" ->
        {:error, "Registry key (intent name) cannot be empty"}

      !is_binary(domain) or String.trim(domain) == "" ->
        {:error, "Registry entry must have a non-empty \"domain\" field"}

      true ->
        :ok
    end
  end

  def validate(:registry_entry, _), do: {:error, "Registry entry must be a {key, value} tuple"}

  def validate(:slot_schema_entry, {key, val}) when is_binary(key) and is_map(val) do
    if String.trim(key) == "" do
      {:error, "Slot schema key cannot be empty"}
    else
      :ok
    end
  end

  def validate(:slot_schema_entry, _), do: {:error, "Slot schema entry must be a {key, value} tuple"}

  def validate(:speech_act_map_entry, {key, val}) when is_binary(key) and is_binary(val) do
    cond do
      String.trim(key) == "" -> {:error, "Speech act map key cannot be empty"}
      String.trim(val) == "" -> {:error, "Speech act map value cannot be empty"}
      true -> :ok
    end
  end

  def validate(:speech_act_map_entry, _), do: {:error, "Speech act map entry must be a {string, string} tuple"}

  def validate(:entity_type_entry, {key, _val}) when is_binary(key) do
    if String.trim(key) == "", do: {:error, "Entity type key cannot be empty"}, else: :ok
  end

  def validate(:entity_type_entry, _), do: {:error, "Entity type entry must be a {key, value} tuple"}

  def validate(:gazetteer_entry, rec) when is_map(rec) do
    value = Map.get(rec, "value") || Map.get(rec, "name") || Map.get(rec, "entry")

    if is_binary(value) and String.trim(value) != "" do
      :ok
    else
      {:error, "Gazetteer entry must have a non-empty value/name/entry field"}
    end
  end

  def validate(:kg_negative, rec) when is_map(rec) do
    head = Map.get(rec, "head")
    relation = Map.get(rec, "relation")
    tail = Map.get(rec, "tail")

    cond do
      !is_binary(head) or String.trim(head) == "" -> {:error, "Must have a non-empty \"head\" field"}
      !is_binary(relation) or String.trim(relation) == "" -> {:error, "Must have a non-empty \"relation\" field"}
      !is_binary(tail) or String.trim(tail) == "" -> {:error, "Must have a non-empty \"tail\" field"}
      true -> :ok
    end
  end

  def validate(:csv_row, _), do: {:error, "CSV rows are read-only"}

  def validate(kind, _), do: {:error, "Unknown record kind: #{inspect(kind)}"}

  @doc "Validate an entire collection before writing."
  @spec validate_all(atom(), list() | map()) :: :ok | {:error, String.t()}
  def validate_all(kind, records) when is_list(records) do
    case Enum.find_value(records, fn rec ->
           case validate(kind, rec) do
             :ok -> nil
             {:error, msg} -> msg
           end
         end) do
      nil -> :ok
      msg -> {:error, msg}
    end
  end

  def validate_all(_kind, %{} = _map), do: :ok
  def validate_all(_, _), do: {:error, "Expected a list or map"}
end
