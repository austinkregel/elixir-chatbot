defmodule Brain.Response.PhraseLatticeRealizer do
  @moduledoc """
  Fragment-level lattice surface realizer.

  Assembles novel responses by selecting and combining typed fragments
  from the PhraseInventory. Each position in the primitive plan gets
  candidate fragments scored by feature vector similarity and tone
  compatibility, then beam search finds the best path considering
  inter-fragment transition smoothness.

  ## Which fragments a position can take

  A fragment is a candidate for a primitive's position only when it can be
  said there as it is written:

    * it is a reply the turn can be given: a fragment of an intent's response
      template (origin `"response_templates"`) is that intent's reply, so it is
      said only for a turn of that intent; one of a domain's response frames
      (origin `"domain_knowledge"`) only for a turn whose intent is of that
      domain; a primitive's phrase (origin `"primitives"`) for any turn;
    * its condition (`Brain.Response.ConditionEvaluator`, such as
      `has_entity:person`) holds for the turn: its entities, slots,
      confidence, speech act and enrichment;
    * every `$placeholder` it names is filled with words: a non-blank string
      or a number from the primitive's content, the enrichment's data, or an
      entity of that type. An atom is a label, never words, so content such as
      `capability: :unknown` fills nothing;
    * it either ends a sentence (terminal punctuation) or leads into the next
      fragment: it ends with a comma ("Okay,"), or it is a hedge or a
      transition ("I think", "Also,"), which qualify or join what follows. A
      fragment that leads in is joined to the next as
      `Brain.Response.Synthesizer.join_after/2` joins a prefix to a sentence,
      so it is never at the plan's last position. A fragment that does neither
      ("Hello") cannot be placed as it is written.

  A plan with a position no fragment can take is not realized:
  `{:error, {:no_fragments, keys}}` names the `type:variant` keys of those
  positions.
  """

  alias Brain.ML.Tokenizer
  alias Brain.Response.{ConditionEvaluator, PhraseInventory, LatticeScorer, TransitionModel, Primitive, Synthesizer}

  require Logger

  @beam_width 5
  @placeholder ~r/\$([a-z][a-z0-9_]*)/

  @doc """
  Realizes a list of content-specified primitives into text via the fragment lattice.

  Options:
    - `:feature_vector` - input feature vector (ChunkFeatures.vector_dimension() dims)
    - `:desired_tone` - 10-dim desired tone vector (from LatticeScorer.compute_desired_tone)
    - `:analysis` - ChunkAnalysis struct; its entities, slots, confidence and
      speech act are what fragment conditions and placeholders are checked against
    - `:analyses` - every chunk's analysis: a primitive whose content names its
      chunk (`:chunk_index`, set by `Brain.Response.DiscoursePlanner`) is worded
      for that chunk's intent, entities and speech act
    - `:unified_context` - context map with entities and enrichment data
    - `:world_id` - the world the response is realized in
    - `:beam_width` - beam width for search (default 5)
    - `:intent` - classified intent for lattice intent bonuses

  Returns `{:ok, rendered, text}` or `{:error, {:no_fragments, keys}}`.
  """
  def realize(primitives, opts \\ []) when is_list(primitives) do
    feature_vector = Keyword.get(opts, :feature_vector, [])
    desired_tone = Keyword.get(opts, :desired_tone, List.duplicate(0.5, 10))
    beam_width = Keyword.get(opts, :beam_width, @beam_width)

    if feature_vector == [] do
      Logger.debug("PhraseLatticeRealizer: empty feature_vector, scoring uses tone and intent only")
    end

    chunks = Enum.map(primitives, &chunk_context(&1, opts))
    candidates_per_position = gather_candidates(primitives, chunks, feature_vector, desired_tone)

    case empty_positions(primitives, candidates_per_position) do
      [] when primitives != [] ->
        path = beam_search(candidates_per_position, beam_width)
        log_selected_path(path)

        texts =
          Enum.zip_with([path, primitives, chunks], fn [{frag, _score}, prim, {_intent, turn}] -> fill(frag.text, prim, turn) end)

        rendered =
          Enum.zip_with(primitives, texts, fn prim, text ->
            prim
            |> Primitive.render(text)
            |> Map.put(:source, :lattice)
          end)

        {:ok, rendered, join_fragments(texts)}

      [] ->
        {:error, {:no_fragments, []}}

      empty ->
        {:error, {:no_fragments, empty}}
    end
  end

  # The intent a primitive is worded for and what its fragments' conditions and
  # placeholders are checked against: those of the chunk it answers (its
  # content's `:chunk_index`, among `opts`' `:analyses`), and of the turn's
  # primary analysis (`:analysis`, `:intent`) for a primitive no chunk is named
  # for. A chunk index no analysis has raises.
  defp chunk_context(%Primitive{content: content}, opts) do
    case Map.fetch(content || %{}, :chunk_index) do
      {:ok, index} ->
        case Enum.find(Keyword.get(opts, :analyses, []), &(Map.get(&1, :chunk_index) == index)) do
          nil ->
            primary = Keyword.get(opts, :analysis)

            if primary && Map.get(primary, :chunk_index) == index do
              {Keyword.get(opts, :intent), turn_context(primary, opts)}
            else
              raise ArgumentError, "PhraseLatticeRealizer: a primitive answers chunk #{inspect(index)}, which no analysis given is"
            end

          chunk ->
            {Map.get(chunk, :intent), turn_context(chunk, opts)}
        end

      :error ->
        {Keyword.get(opts, :intent), turn_context(Keyword.get(opts, :analysis) || %{}, opts)}
    end
  end

  defp turn_context(analysis, opts) do
    unified_context = Keyword.get(opts, :unified_context, %{}) || %{}
    enrichment = Map.get(unified_context, :enrichment) || %{}
    slots = Map.get(analysis, :slots)

    %{
      entities: Map.get(analysis, :entities) || [],
      filled_slots: filled_slot_names(slots),
      missing_slots: (slots && Map.get(slots, :missing_required)) || [],
      confidence: Map.get(analysis, :confidence) || 0.0,
      speech_act: Map.get(analysis, :speech_act) || %{},
      enriched_data: Map.get(enrichment, :enriched_data) || %{},
      enrichment_status: Map.get(enrichment, :enrichment_status),
      world_id: Keyword.get(opts, :world_id)
    }
  end

  defp filled_slot_names(%{filled_slots: filled}) when is_map(filled), do: filled |> Map.keys() |> Enum.map(&to_string/1)
  defp filled_slot_names(_), do: []

  defp gather_candidates(primitives, chunks, feature_vector, desired_tone) do
    last = length(primitives) - 1

    [primitives, chunks, Enum.to_list(0..last//1)]
    |> Enum.zip_with(fn [prim, {intent, turn}, index] -> {prim, intent, turn, index} end)
    |> Enum.map(fn {%Primitive{type: type, variant: variant} = prim, intent, turn, index} ->
      scorer_opts = if intent, do: [intent: intent], else: []

      fragments =
        if PhraseInventory.ready?() do
          type
          |> PhraseInventory.lookup_by_primitive(variant)
          |> Enum.filter(&sayable_here?(&1, prim, turn, intent, index == last))
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

  defp sayable_here?(frag, prim, turn, intent, last?) do
    replies_to?(frag, intent) and
      ConditionEvaluator.evaluate(frag.condition, turn) and
      Enum.all?(placeholders(frag.text), &(fill_value(&1, prim, turn) != nil)) and
      placeable?(frag, prim, last?)
  end

  defp placeable?(frag, prim, last?) do
    Tokenizer.ends_with_terminal_punctuation?(frag.text) or (not last? and leads_in?(frag.text, prim))
  end

  defp leads_in?(text, %Primitive{type: type}) do
    String.ends_with?(String.trim_trailing(text), ",") or type in [:hedging, :transition]
  end

  defp replies_to?(%{origin: "response_templates", source_intent: source}, intent), do: source == intent

  defp replies_to?(%{origin: "domain_knowledge", source_intent: source}, intent),
    do: is_binary(intent) and domain(source) == domain(intent)

  defp replies_to?(_frag, _intent), do: true

  defp domain(label), do: label |> String.split(".", parts: 2) |> hd()

  defp empty_positions(primitives, candidates_per_position) do
    primitives
    |> Enum.zip(candidates_per_position)
    |> Enum.filter(fn {_prim, candidates} -> candidates == [] end)
    |> Enum.map(fn {%Primitive{type: type, variant: variant}, _} -> "#{type}:#{variant || "_"}" end)
  end

  defp placeholders(text), do: @placeholder |> Regex.scan(text, capture: :all_but_first) |> List.flatten()

  # The words a placeholder is filled with: the primitive's content, then the
  # enrichment's data, then an entity of that type; nil when none says it.
  defp fill_value(name, %Primitive{content: content}, turn) do
    words(lookup(content, name)) || words(lookup(turn.enriched_data, name)) || entity_words(turn.entities, name)
  end

  defp lookup(map, name) when is_map(map) do
    Enum.find_value(map, fn {key, value} -> if to_string(key) == name, do: value end)
  end

  defp lookup(_, _name), do: nil

  defp entity_words(entities, type) do
    Enum.find_value(entities, fn entity ->
      if to_string(Map.get(entity, :entity_type)) == type, do: words(Map.get(entity, :value))
    end)
  end

  defp words(value) when is_binary(value), do: if(String.trim(value) == "", do: nil, else: value)
  defp words(value) when is_number(value), do: to_string(value)
  defp words(_value), do: nil

  defp fill(text, prim, turn) do
    Regex.replace(@placeholder, text, fn _placeholder, name -> fill_value(name, prim, turn) end)
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

    [{best_path, _score} | _] = final_paths
    best_path
  end

  # A fragment that ends a sentence is followed by the next as a new sentence;
  # one that does not leads into it.
  defp join_fragments(texts) do
    Enum.reduce(texts, "", fn text, acc ->
      cond do
        acc == "" -> text
        Tokenizer.ends_with_terminal_punctuation?(acc) -> acc <> " " <> text
        true -> Synthesizer.join_after(acc, text)
      end
    end)
  end
end
