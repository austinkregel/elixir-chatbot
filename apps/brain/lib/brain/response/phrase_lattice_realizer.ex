defmodule Brain.Response.PhraseLatticeRealizer do
  @moduledoc """
  Fragment-level lattice surface realizer.

  Assembles novel responses by selecting and combining typed fragments
  from the PhraseInventory. Each position in the primitive plan gets
  candidate fragments scored by feature vector similarity and tone
  compatibility, then beam search finds the best path considering
  inter-fragment transition smoothness.
  """

  alias Brain.Response.{PhraseInventory, LatticeScorer, TransitionModel, Primitive}

  require Logger

  @beam_width 5
  @min_fragments_for_lattice 2

  @doc """
  Realizes a list of content-specified primitives into text via the fragment lattice.

  Options:
    - `:feature_vector` - input feature vector (ChunkFeatures.vector_dimension() dims)
    - `:desired_tone` - 10-dim desired tone vector (from LatticeScorer.compute_desired_tone)
    - `:analysis` - ChunkAnalysis struct
    - `:unified_context` - context map with entities and enrichment data
    - `:beam_width` - beam width for search (default 5)
    - `:intent` - classified intent for lattice intent bonuses
  """
  def realize(primitives, opts \\ []) when is_list(primitives) do
    feature_vector = Keyword.get(opts, :feature_vector, [])
    desired_tone = Keyword.get(opts, :desired_tone, List.duplicate(0.5, 10))
    beam_width = Keyword.get(opts, :beam_width, @beam_width)
    unified_context = Keyword.get(opts, :unified_context, %{})
    intent = Keyword.get(opts, :intent)

    if feature_vector == [] do
      Logger.debug("PhraseLatticeRealizer: empty feature_vector, scoring uses tone and intent only")
    end

    candidates_per_position = gather_candidates(primitives, feature_vector, desired_tone, intent)

    if all_positions_have_candidates?(candidates_per_position) do
      path = beam_search(candidates_per_position, beam_width)
      log_selected_path(path)
      text = assemble_text(path, primitives, unified_context)

      rendered =
        Enum.zip(primitives, path)
        |> Enum.map(fn {prim, {frag, _score}} ->
          prim
          |> Primitive.render(frag.text)
          |> Map.put(:source, :lattice)
        end)

      {:ok, rendered, text}
    else
      {:error, :insufficient_inventory}
    end
  end

  defp gather_candidates(primitives, feature_vector, desired_tone, intent) do
    scorer_opts = if intent, do: [intent: intent], else: []

    Enum.map(primitives, fn %Primitive{type: type, variant: variant} ->
      fragments =
        if PhraseInventory.ready?() do
          PhraseInventory.lookup_by_primitive(type, variant)
        else
          []
        end

      if fragments == [] do
        []
      else
        LatticeScorer.score_fragments(fragments, feature_vector, desired_tone, scorer_opts)
      end
    end)
  end

  defp log_selected_path(path) do
    ids =
      path
      |> Enum.map(fn {%{id: id}, _score} -> id end)
      |> Enum.reject(&is_nil/1)

    if ids != [] do
      Logger.debug("PhraseLatticeRealizer: selected fragment ids #{inspect(ids)}")
    end
  end

  defp all_positions_have_candidates?(candidates_per_position) do
    length(candidates_per_position) >= @min_fragments_for_lattice and
      Enum.all?(candidates_per_position, fn candidates -> candidates != [] end)
  end

  defp beam_search(candidates_per_position, beam_width) do
    first_candidates = hd(candidates_per_position)
    initial_beams = Enum.take(first_candidates, beam_width)
    initial_paths = Enum.map(initial_beams, fn {frag, score} -> {[{frag, score}], score} end)

    rest_positions = tl(candidates_per_position)

    final_paths =
      Enum.reduce(rest_positions, initial_paths, fn position_candidates, beams ->
        expanded =
          for {path, path_score} <- beams,
              {frag, frag_score} <- Enum.take(position_candidates, beam_width * 2) do
            {prev_frag, _} = List.last(path)
            transition = TransitionModel.score(prev_frag.text, frag.text)
            combined = path_score + frag_score * 0.6 + transition * 0.4
            {path ++ [{frag, frag_score}], combined}
          end

        expanded
        |> Enum.sort_by(fn {_path, score} -> score end, :desc)
        |> Enum.take(beam_width)
      end)

    case final_paths do
      [{best_path, _score} | _] -> best_path
      [] -> []
    end
  end

  defp assemble_text(path, primitives, unified_context) do
    texts =
      Enum.zip(path, primitives)
      |> Enum.map(fn {{frag, _score}, prim} ->
        text = frag.text
        text = substitute_primitive_content(text, prim)
        substitute_enrichment(text, unified_context)
      end)

    join_fragments(texts)
  end

  defp substitute_primitive_content(text, %Primitive{content: content}) when is_map(content) do
    Enum.reduce(content, text, fn {key, value}, acc ->
      placeholder = "$#{key}"
      value_str = to_string_safe(value)

      if String.contains?(acc, placeholder) do
        String.replace(acc, placeholder, value_str)
      else
        acc
      end
    end)
  end

  defp substitute_primitive_content(text, _), do: text

  defp substitute_enrichment(text, %{enrichment: %{enriched_data: data}}) when is_map(data) do
    Enum.reduce(data, text, fn {key, value}, acc ->
      placeholder = "$#{key}"
      value_str = to_string_safe(value)

      if String.contains?(acc, placeholder) do
        String.replace(acc, placeholder, value_str)
      else
        acc
      end
    end)
  end

  defp substitute_enrichment(text, _), do: text

  defp join_fragments(texts) do
    texts
    |> Enum.filter(&(String.trim(&1) != ""))
    |> Enum.join(" ")
    |> String.trim()
  end

  defp to_string_safe(value) when is_binary(value), do: value
  defp to_string_safe(value) when is_atom(value), do: Atom.to_string(value)
  defp to_string_safe(value) when is_number(value), do: to_string(value)
  defp to_string_safe(value), do: inspect(value)
end
