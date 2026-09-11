defmodule Brain.Response.ChunkSegmenter do
  @moduledoc "Segments response templates into typed chunks for template blending.\n\nUses sentence boundaries and embedding-based classification to identify\nchunk types. Chunks can then be recombined to generate novel responses.\n\n## Chunk Types\n\n- `:greeting` - Opening phrases (\"Hello!\", \"Nice to meet you!\")\n- `:acknowledgment` - Confirmation phrases (\"I understand.\", \"Got it.\")\n- `:body` - Substantive content (\"The weather in $location is...\")\n- `:offer` - Invitations for next action (\"What can I help with?\")\n- `:clarification` - Requests for missing info (\"Which location?\")\n- `:closing` - Farewell phrases (\"Have a great day!\", \"Goodbye!\")\n\n## Usage\n\n    ChunkSegmenter.segment(\"Hello! The weather is sunny. Anything else?\")\n    # => [\n    #   %Chunk{text: \"Hello!\", type: :greeting},\n    #   %Chunk{text: \"The weather is sunny.\", type: :body},\n    #   %Chunk{text: \"Anything else?\", type: :offer}\n    # ]\n"

  alias Brain.Memory.Embedder
  alias Brain.ML.Tokenizer

  require Logger

  defmodule Chunk do
    @moduledoc "A segmented piece of a response template"
    defstruct [
      :text,
      :type,
      :embedding,
      :source_intent,
      :tone,
      :tone_vector,
      :prototype_vector,
      :primitive_type,
      :primitive_variant,
      slots: [],
      enrichment_fields: [],
      conditions: %{}
    ]
  end

  @smalltalk_path "priv/knowledge/domains/smalltalk.json"
  @external_resource @smalltalk_path

  @chunk_type_seeds (case File.read(@smalltalk_path) do
                       {:ok, content} ->
                         case Jason.decode(content) do
                           {:ok, data} ->
                             seeds = Map.get(data, "chunk_type_seeds", %{})

                             %{
                               greeting: Map.get(seeds, "greeting", []),
                               acknowledgment: Map.get(seeds, "acknowledgment", []),
                               body: Map.get(seeds, "body", []),
                               offer: Map.get(seeds, "offer", []),
                               clarification: Map.get(seeds, "clarification", []),
                               closing: Map.get(seeds, "closing", [])
                             }

                           {:error, _} ->
                             %{
                               greeting: [],
                               acknowledgment: [],
                               body: [],
                               offer: [],
                               clarification: [],
                               closing: []
                             }
                         end

                       {:error, _} ->
                         %{
                           greeting: [],
                           acknowledgment: [],
                           body: [],
                           offer: [],
                           clarification: [],
                           closing: []
                         }
                     end)
  @centroid_key :chunk_type_centroids

  @doc "Segments a template text into typed chunks.\n\nReturns a list of %Chunk{} structs, each with:\n- text: The chunk text\n- type: The classified chunk type (:greeting, :body, etc.)\n- embedding: TF-IDF embedding for the chunk\n"
  def segment(template_text) when is_binary(template_text) do
    template_text
    |> split_into_sentences()
    |> Enum.map(&classify_and_embed/1)
    |> Enum.filter(& &1)
  end

  @doc "Segments a template and associates it with an intent.\n"
  def segment(template_text, source_intent) when is_binary(template_text) do
    template_text
    |> segment()
    |> Enum.map(fn chunk -> %{chunk | source_intent: source_intent} end)
  end

  @doc "Segments all templates from a map of {intent => [template_texts]}.\nReturns a flat list of all chunks.\n"
  def segment_all(templates_by_intent) when is_map(templates_by_intent) do
    templates_by_intent
    |> Enum.flat_map(fn {intent, templates} ->
      Enum.flat_map(templates, fn template ->
        segment(template, intent)
      end)
    end)
  end

  @doc """
  Segments a template text into typed chunks, preserving $placeholder slot tokens.

  Extracts slot names and enrichment fields from $placeholder patterns before
  splitting, then attaches them to each resulting chunk.
  """
  def segment_with_slots(template_text) when is_binary(template_text) do
    {slots, enrichment_fields} = extract_slot_info(template_text)

    template_text
    |> split_into_sentences()
    |> Enum.map(fn sentence ->
      chunk = classify_and_embed(sentence)
      %{chunk | slots: slots, enrichment_fields: enrichment_fields}
    end)
    |> Enum.filter(& &1)
  end

  @doc """
  Segments a template with slots and associates it with an intent.
  """
  def segment_with_slots(template_text, source_intent) when is_binary(template_text) do
    template_text
    |> segment_with_slots()
    |> Enum.map(fn chunk -> %{chunk | source_intent: source_intent} end)
  end

  @entity_slots MapSet.new(~w(location device user name city state country date time))

  defp extract_slot_info(text) do
    tokens = Tokenizer.tokenize(text)

    slot_names =
      tokens
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [t, _next] -> t.text == "$" end)
      |> Enum.map(fn [_dollar, name_token] -> name_token.normalized || name_token.text end)

    enrichment_fields =
      Enum.reject(slot_names, fn name -> MapSet.member?(@entity_slots, name) end)

    {slot_names, enrichment_fields}
  end

  @hedge_words MapSet.new(~w(maybe perhaps possibly might could probably seemingly apparently
                             arguably presumably supposedly potentially conceivably))
  @formal_words MapSet.new(~w(furthermore moreover consequently nevertheless accordingly
                              henceforth whereas hereby therein notwithstanding
                              pursuant regarding respectfully))

  @doc """
  Estimates the tone of a chunk based on lexical signals.

  Populates the `:tone` and `:tone_vector` fields. The tone_vector is a
  10-dimensional vector: [pos, neg, neu, conf, polarity, arousal, warmth,
  formality, playfulness, directness].
  """
  def tag_tone(%Chunk{text: text} = chunk) when is_binary(text) do
    tokens = Tokenizer.tokenize(text)
    normalized = Enum.map(tokens, fn t -> t.normalized || t.text end)
    token_set = MapSet.new(normalized)
    count = max(length(normalized), 1)

    exclamation_count =
      Enum.count(tokens, fn t -> t.text == "!" end)

    hedge_count =
      Enum.count(normalized, fn t -> MapSet.member?(@hedge_words, t) end)

    formal_count =
      Enum.count(normalized, fn t -> MapSet.member?(@formal_words, t) end)

    has_question = MapSet.member?(token_set, "?")

    arousal = min(1.0, 0.3 + exclamation_count * 0.2)
    formality = min(1.0, 0.3 + formal_count / count * 2.0)
    directness = max(0.0, 0.7 - hedge_count / count * 2.0)
    warmth = if exclamation_count > 0, do: 0.6, else: 0.4
    playfulness = if exclamation_count > 1, do: 0.5, else: 0.2
    pos = min(1.0, 0.3 + exclamation_count * 0.1)
    neg = if has_question and hedge_count > 0, do: 0.3, else: 0.1
    neu = max(0.0, 1.0 - pos - neg)
    conf = max(0.0, directness - hedge_count * 0.1)
    polarity = pos - neg

    tone_vector = [
      pos,
      neg,
      neu,
      conf,
      polarity,
      arousal,
      warmth,
      formality,
      playfulness,
      directness
    ]

    tone =
      cond do
        formality > 0.6 -> :formal
        arousal > 0.6 -> :enthusiastic
        directness < 0.4 -> :hedging
        warmth > 0.5 -> :warm
        true -> :neutral
      end

    %{chunk | tone: tone, tone_vector: tone_vector}
  end

  def tag_tone(%Chunk{} = chunk), do: chunk

  @doc "Returns the chunk type seeds used for classification.\n"
  def get_type_seeds do
    @chunk_type_seeds
  end

  @doc "Clears the cached centroids (useful for testing).\n"
  def clear_centroids do
    Process.delete(@centroid_key)
    :ok
  end

  @doc "Splits text into sentences using punctuation boundaries.\n"
  def split_into_sentences(text) when is_binary(text) do
    text
    |> Tokenizer.split_sentences()
    |> Enum.map(& &1.text)
    |> Enum.map(&String.trim/1)
    |> Enum.filter(&(String.length(&1) > 0))
  end

  defp classify_and_embed(sentence) do
    heuristic_type = classify_by_heuristic(sentence)

    embedding =
      case Embedder.embed(sentence) do
        {:ok, emb} -> emb
        _ -> nil
      end

    chunk_type =
      if heuristic_type != :body do
        heuristic_type
      else
        if embedding do
          classify_chunk_type(embedding)
        else
          :body
        end
      end

    %Chunk{
      text: sentence,
      type: chunk_type,
      embedding: embedding,
      source_intent: nil
    }
  end

  defp classify_chunk_type(embedding) do
    centroids = get_or_build_centroids()

    {best_type, _best_similarity} =
      centroids
      |> Enum.map(fn {type, centroid} ->
        similarity = cosine_similarity(embedding, centroid)
        {type, similarity}
      end)
      |> Enum.max_by(fn {_type, sim} -> sim end, fn -> {:body, 0.0} end)

    best_type
  end

  @greeting_tokens MapSet.new(~w(hello hi hey welcome howdy greetings))
  @closing_tokens MapSet.new(~w(goodbye bye farewell))
  @offer_tokens MapSet.new(~w(help assist anything))
  @clarification_start_tokens MapSet.new(~w(which what where when how))
  @acknowledgment_tokens MapSet.new(~w(okay sure understood acknowledged right))

  defp classify_by_heuristic(sentence) do
    raw_tokens = Tokenizer.tokenize(sentence)
    tokens = Enum.map(raw_tokens, fn t -> t.normalized || t.text end)
    token_set = MapSet.new(tokens)
    first_token = List.first(tokens)

    cond do
      first_token in @greeting_tokens or
        (first_token == "good" and Enum.at(tokens, 1) in ~w(morning afternoon evening)) or
          (first_token == "nice" and Enum.at(tokens, 1) == "to" and Enum.at(tokens, 2) == "meet") ->
        :greeting

      first_token in @closing_tokens or
        (first_token == "take" and Enum.at(tokens, 1) == "care") or
        (first_token == "see" and Enum.at(tokens, 1) == "you") or
          (MapSet.member?(token_set, "great") and MapSet.member?(token_set, "day")) ->
        :closing

      not MapSet.disjoint?(token_set, @offer_tokens) and
          (MapSet.member?(token_set, "else") or MapSet.member?(token_set, "can") or
             MapSet.member?(token_set, "would")) ->
        :offer

      first_token in @clarification_start_tokens or
        (MapSet.member?(token_set, "need") and MapSet.member?(token_set, "know")) or
          (List.last(tokens) == "?" and Enum.count_until(tokens, 8) < 8) ->
        :clarification

      (first_token == "i" and Enum.at(tokens, 1) == "understand") or
        (first_token == "got" and Enum.at(tokens, 1) == "it") or
          first_token in @acknowledgment_tokens ->
        :acknowledgment

      true ->
        :body
    end
  end

  defp get_or_build_centroids do
    case Process.get(@centroid_key) do
      nil ->
        centroids = build_centroids()
        Process.put(@centroid_key, centroids)
        centroids

      centroids ->
        centroids
    end
  end

  defp build_centroids do
    if Embedder.ready?() do
      @chunk_type_seeds
      |> Enum.map(fn {type, seeds} ->
        embeddings =
          seeds
          |> Enum.map(fn seed ->
            case Embedder.embed(seed) do
              {:ok, embedding} -> embedding
              _ -> nil
            end
          end)
          |> Enum.filter(& &1)

        centroid =
          if embeddings != [] do
            average_vectors(embeddings)
          else
            nil
          end

        {type, centroid}
      end)
      |> Enum.filter(fn {_, centroid} -> centroid != nil end)
      |> Map.new()
    else
      %{}
    end
  end

  defp average_vectors(vectors) when is_list(vectors) and vectors != [] do
    n = length(vectors)
    vec_length = length(List.first(vectors))

    summed =
      Enum.reduce(vectors, List.duplicate(0.0, vec_length), fn vec, acc ->
        Enum.zip(vec, acc)
        |> Enum.map(fn {a, b} -> a + b end)
      end)

    Enum.map(summed, fn x -> x / n end)
  end

  defp average_vectors(_) do
    nil
  end

  defp cosine_similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    if length(vec1) != length(vec2) do
      0.0
    else
      dot = Enum.zip(vec1, vec2) |> Enum.reduce(0.0, fn {a, b}, sum -> sum + a * b end)
      mag1 = :math.sqrt(Enum.reduce(vec1, 0.0, fn x, sum -> sum + x * x end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0.0, fn x, sum -> sum + x * x end))

      if mag1 == 0.0 or mag2 == 0.0 do
        0.0
      else
        dot / (mag1 * mag2)
      end
    end
  end

  defp cosine_similarity(_, _) do
    0.0
  end
end
