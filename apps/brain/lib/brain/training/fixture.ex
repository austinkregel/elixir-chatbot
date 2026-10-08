defmodule Brain.Training.Fixture do
  @moduledoc """
  Loads and validates training fixtures in the task 086 format.

  A fixture file is `{"format_version": 1, "records": [...]}`. Every record
  carries its identity, provenance and annotation layers:

      %{
        "id" => "ud_ewt/3f2a…", "text" => "…",
        "origin" => "corpus:ud_ewt", "source_id" => "…", "license" => "cc-by-sa-4.0",
        "produced_at" => "2026-09-19T…Z",
        "producer" => %{"name" => …, "version" => …,
                        "inputs" => [%{"name" => …, "version" => …, "sha256" => …}]},
        "tokens" => [...],
        "layers" => %{"pos" => %{"tags" => [...], "vocabulary" => "ud_v1",
                                 "resolution" => [...]}}
      }

  A file is validated whole before anything is returned, and any breach of
  the rules fails loudly, naming the file and the record: a missing field, a
  layer whose length differs from the tokens, a tag outside the declared
  vocabulary, an unknown resolution tier, or a repeated id. Nothing is
  filtered out or defaulted -- that is how 4,870 gold-standard records once
  became zero training sequences without a word.
  """

  @format_version 1

  @required ~w(id text origin source_id license produced_at producer tokens layers)

  # Which tier produced each tag. "corpus" is a label annotated in the source
  # corpus itself; the others are the WordNet generator's tiers (task 086).
  @resolutions ~w(corpus closed_class wordnet morphology authored)

  @doc "The annotation vocabularies a layer may declare, with their tags."
  @spec vocabularies() :: %{String.t() => [String.t()]}
  def vocabularies, do: %{"ud_v1" => Brain.ML.POSTagger.valid_tags()}

  @doc """
  Reads and validates a fixture file. Returns its records, or raises naming
  the first rule broken.
  """
  @spec load!(Path.t()) :: [map()]
  def load!(path) do
    data = path |> File.read!() |> Jason.decode!()

    records =
      case data do
        %{"format_version" => @format_version, "records" => records} when is_list(records) ->
          records

        %{"format_version" => other} ->
          fail(path, nil, "format_version #{inspect(other)}, expected #{@format_version}")

        _ ->
          fail(path, nil, "not a fixture file: expected {\"format_version\", \"records\"}")
      end

    if records == [], do: fail(path, nil, "holds no records")

    Enum.each(records, &validate!(&1, path))

    duplicates = records |> Enum.frequencies_by(& &1["id"]) |> Enum.filter(fn {_id, n} -> n > 1 end)
    if duplicates != [], do: fail(path, nil, "repeated ids: #{inspect(Enum.map(duplicates, &elem(&1, 0)))}")

    records
  end

  @doc """
  The POS layer of each record as a `%{tokens: [...], tags: [...]}` sequence,
  the shape `Brain.ML.POSTagger.train/1` reads. Raises if a record has no POS
  layer.
  """
  @spec pos_sequences([map()]) :: [%{tokens: [String.t()], tags: [String.t()]}]
  def pos_sequences(records) do
    Enum.map(records, fn record ->
      case record["layers"] do
        %{"pos" => %{"tags" => tags}} -> %{tokens: record["tokens"], tags: tags}
        _ -> raise ArgumentError, "Fixture: record #{record["id"]} has no pos layer"
      end
    end)
  end

  defp validate!(record, path) when is_map(record) do
    id = record["id"]

    for field <- @required, not Map.has_key?(record, field) do
      fail(path, id, "missing required field #{inspect(field)}")
    end

    for field <- ~w(id text origin source_id license produced_at),
        not (is_binary(record[field]) and record[field] != "") do
      fail(path, id, "#{field} must be a non-empty string, got #{inspect(record[field])}")
    end

    case DateTime.from_iso8601(record["produced_at"]) do
      {:ok, _, _} -> :ok
      _ -> fail(path, id, "produced_at is not an ISO 8601 datetime: #{inspect(record["produced_at"])}")
    end

    validate_producer!(record["producer"], path, id)

    tokens = record["tokens"]

    unless is_list(tokens) and tokens != [] and Enum.all?(tokens, &(is_binary(&1) and &1 != "")) do
      fail(path, id, "tokens must be a non-empty list of non-empty strings")
    end

    case record["layers"] do
      layers when is_map(layers) and map_size(layers) > 0 ->
        Enum.each(layers, fn {name, layer} -> validate_layer!(name, layer, length(tokens), path, id) end)

      other ->
        fail(path, id, "layers must be a non-empty object, got #{inspect(other)}")
    end
  end

  defp validate!(record, path), do: fail(path, nil, "a record is not an object: #{inspect(record)}")

  defp validate_producer!(%{"name" => name, "version" => version, "inputs" => [_ | _] = inputs}, path, id)
       when is_binary(name) and is_binary(version) do
    for input <- inputs do
      case input do
        %{"name" => n, "version" => v, "sha256" => sha}
        when is_binary(n) and is_binary(v) and is_binary(sha) and byte_size(sha) == 64 ->
          :ok

        other ->
          fail(path, id, "producer input needs name, version and a 64-hex sha256: #{inspect(other)}")
      end
    end
  end

  defp validate_producer!(other, path, id),
    do: fail(path, id, "producer needs name, version and at least one input: #{inspect(other)}")

  defp validate_layer!(name, %{"tags" => tags, "vocabulary" => vocabulary, "resolution" => resolution}, n, path, id)
       when is_list(tags) and is_list(resolution) do
    allowed =
      case Map.fetch(vocabularies(), vocabulary) do
        {:ok, tagset} -> tagset
        :error -> fail(path, id, "layer #{name}: unknown vocabulary #{inspect(vocabulary)}")
      end

    if length(tags) != n, do: fail(path, id, "layer #{name}: #{length(tags)} tags for #{n} tokens")
    if length(resolution) != n, do: fail(path, id, "layer #{name}: #{length(resolution)} resolutions for #{n} tokens")

    case Enum.reject(tags, &(&1 in allowed)) do
      [] -> :ok
      bad -> fail(path, id, "layer #{name}: tags outside #{vocabulary}: #{inspect(Enum.uniq(bad))}")
    end

    case Enum.reject(resolution, &(&1 in @resolutions)) do
      [] -> :ok
      bad -> fail(path, id, "layer #{name}: unknown resolution tiers #{inspect(Enum.uniq(bad))}")
    end
  end

  defp validate_layer!(name, other, _n, path, id),
    do: fail(path, id, "layer #{name} needs tags, vocabulary and resolution: #{inspect(other)}")

  defp fail(path, nil, message), do: raise("Fixture #{path}: #{message}")
  defp fail(path, id, message), do: raise("Fixture #{path}, record #{inspect(id)}: #{message}")
end
