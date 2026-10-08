defmodule Brain.Lexicon.TypeAnchors do
  @moduledoc """
  Maps WordNet senses to entity types through declared anchor synsets.

  The anchors live in `priv/analysis/wordnet_type_anchors.json`. A sense
  belongs to the type of the nearest anchor among its ancestors, so a singer
  -- reached through "musician" before "person" -- is a `music_artist`, while
  a mother is a `person`.

  Used at seed time only: `Brain.Lexicon.Seeder` classifies each SemCor-counted
  sense once and writes the result into the brain's own lexicon. Nothing
  classifies senses against WordNet at runtime.
  """

  alias Brain.ML.Lexicon, as: WordNet

  @anchors_path Path.join(:code.priv_dir(:brain), "analysis/wordnet_type_anchors.json")
  @external_resource @anchors_path

  @doc """
  Loads the anchor table as `%{synset_id => entity_type}`.

  Raises if the file is missing or malformed, if an anchor's synset is not in
  WordNet, if that synset no longer contains the word the table names for it,
  or if one synset is claimed by two types.
  """
  @spec load!(keyword()) :: %{integer() => String.t()}
  def load!(opts \\ []) do
    path = Keyword.get(opts, :path, @anchors_path)
    lexicon = Keyword.get(opts, :lexicon, WordNet)

    anchors =
      case path |> File.read!() |> Jason.decode!() do
        %{"anchors" => anchors} when is_map(anchors) and map_size(anchors) > 0 -> anchors
        _ -> raise "TypeAnchors: #{path} has no \"anchors\" object"
      end

    Enum.reduce(anchors, %{}, fn {type, entries}, acc ->
      Enum.reduce(entries, acc, fn entry, acc ->
        synset_id = verify_anchor!(type, entry, lexicon, path)

        case Map.fetch(acc, synset_id) do
          {:ok, other} ->
            raise "TypeAnchors: synset #{synset_id} is anchored to both " <>
                    "#{inspect(other)} and #{inspect(type)} in #{path}"

          :error ->
            Map.put(acc, synset_id, type)
        end
      end)
    end)
  end

  defp verify_anchor!(type, %{"synset_id" => synset_id, "word" => word}, lexicon, path)
       when is_integer(synset_id) and is_binary(word) do
    case WordNet.synset(synset_id, lexicon) do
      {:ok, %{words: words}} ->
        if word in words do
          synset_id
        else
          raise "TypeAnchors: #{inspect(type)} anchor #{synset_id} should contain " <>
                  "#{inspect(word)} but WordNet has #{inspect(words)} (#{path})"
        end

      :error ->
        raise "TypeAnchors: #{inspect(type)} anchor #{synset_id} is not a WordNet synset (#{path})"
    end
  end

  defp verify_anchor!(type, entry, _lexicon, path) do
    raise "TypeAnchors: #{inspect(type)} has a malformed anchor #{inspect(entry)} in #{path}; " <>
            "expected {\"synset_id\": integer, \"word\": string}"
  end

  @doc """
  Returns the entity type of the nearest anchor at or above `synset_id`, or
  `:unanchored` when no ancestor is anchored.

  Raises when two different types are anchored at the same nearest distance:
  the sense would belong to both, and the table must say which it means.
  """
  @spec type_of_sense(integer(), %{integer() => String.t()}, keyword()) ::
          {:ok, String.t()} | :unanchored
  def type_of_sense(synset_id, anchors, opts \\ []) do
    lexicon = Keyword.get(opts, :lexicon, WordNet)

    case Map.fetch(anchors, synset_id) do
      {:ok, type} ->
        {:ok, type}

      :error ->
        synset_id
        |> WordNet.synset_ancestors(name: lexicon)
        |> Enum.filter(fn {sid, _words, _distance} -> Map.has_key?(anchors, sid) end)
        |> nearest_type(synset_id, anchors)
    end
  end

  defp nearest_type([], _synset_id, _anchors), do: :unanchored

  defp nearest_type([{_sid, _words, distance} | _] = hits, synset_id, anchors) do
    types =
      hits
      |> Enum.filter(fn {_sid, _words, d} -> d == distance end)
      |> Enum.map(fn {sid, _words, _d} -> Map.fetch!(anchors, sid) end)
      |> Enum.uniq()

    case types do
      [type] ->
        {:ok, type}

      _ ->
        raise "TypeAnchors: synset #{synset_id} reaches anchors of #{inspect(types)} " <>
                "at the same distance (#{distance}); the anchor table must disambiguate it"
    end
  end
end
