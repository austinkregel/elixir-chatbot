defmodule Brain.Response.ChunkSegmenter do
  @moduledoc "Segments response templates into typed chunks for template blending.\n\nUses sentence boundaries and embedding-based classification to identify\nchunk types. Chunks can then be recombined to generate novel responses.\n\nChunks are embedded, and classified against the chunk type centroids, by a\nworld's embedder (`Brain.Worlds.Embedder`), so every segmenting call names its world.\nA world whose embedder cannot embed raises. While its vocabulary is being\nrebuilt from its episodes, segmenting waits for the rebuild to finish\n(`Brain.Worlds.Embedder.await_ready/2`); the `try_` functions, for a process that a\nrebuild calls, return `{:error, :vocabulary_building}` instead. A world's centroids are built once and kept until its\nvocabulary is rebuilt or it is destroyed (`forget_world/1`); a chunk embedding\nwhose length differs from the centroids' raises.\n\n## Chunk Types\n\n- `:greeting` - Opening phrases (\"Hello!\", \"Nice to meet you!\")\n- `:acknowledgment` - Confirmation phrases (\"I understand.\", \"Got it.\")\n- `:body` - Substantive content (\"The weather in $location is...\")\n- `:offer` - Invitations for next action (\"What can I help with?\")\n- `:clarification` - Requests for missing info (\"Which location?\")\n- `:closing` - Farewell phrases (\"Have a great day!\", \"Goodbye!\")\n\n## Usage\n\n    ChunkSegmenter.segment(world_id, \"Hello! The weather is sunny. Anything else?\")\n    # => [\n    #   %Chunk{text: \"Hello!\", type: :greeting},\n    #   %Chunk{text: \"The weather is sunny.\", type: :body},\n    #   %Chunk{text: \"Anything else?\", type: :offer}\n    # ]\n"

  alias Brain.Worlds.Embedder, as: WorldEmbedder
  alias Brain.ML.Tokenizer

  require Logger

  defmodule Chunk do
    @moduledoc "A segmented piece of a response template"
    defstruct [:text, :type, :embedding, :source_intent, :tone, :tone_vector,
               :prototype_vector, :primitive_type, :primitive_variant,
               slots: [], enrichment_fields: [], conditions: %{}]
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
                             %{greeting: [], acknowledgment: [], body: [],
                               offer: [], clarification: [], closing: []}
                         end

                       {:error, _} ->
                         %{greeting: [], acknowledgment: [], body: [],
                           offer: [], clarification: [], closing: []}
                     end)
  @centroid_key :chunk_type_centroids

  @doc "Segments a template text into typed chunks, in the world's embedder.\n\nReturns a list of %Chunk{} structs, each with:\n- text: The chunk text\n- type: The classified chunk type (:greeting, :body, etc.)\n- embedding: the world's TF-IDF embedding for the chunk\n\nWhile the world's vocabulary is being rebuilt this waits for the rebuild to\nfinish (`Brain.Worlds.Embedder.await_ready/2`).\n"
  def segment(world_id, template_text) when is_binary(world_id) and is_binary(template_text) do
    awaiting_rebuild(world_id, fn -> try_segment(world_id, template_text) end)
  end

  @doc "Segments a template in the world's embedder and associates it with an intent.\n"
  def segment(world_id, template_text, source_intent)
      when is_binary(world_id) and is_binary(template_text) do
    awaiting_rebuild(world_id, fn -> try_segment(world_id, template_text, source_intent) end)
  end

  @doc "Segments all templates from a map of {intent => [template_texts]} in the\nworld's embedder. Returns a flat list of all chunks.\n"
  def segment_all(world_id, templates_by_intent) when is_binary(world_id) and is_map(templates_by_intent) do
    awaiting_rebuild(world_id, fn -> try_segment_all(world_id, templates_by_intent) end)
  end

  @doc """
  `segment/2` that does not wait for a rebuild: `{:ok, chunks}`, or
  `{:error, :vocabulary_building}` while the world's vocabulary is being
  rebuilt. For a process that a rebuild calls, which must not wait in its own
  callbacks (`Brain.Worlds.Embedder`): it returns the error to its caller, which waits.
  """
  @spec try_segment(String.t(), String.t()) :: {:ok, [%Chunk{}]} | {:error, :vocabulary_building}
  def try_segment(world_id, template_text) when is_binary(world_id) and is_binary(template_text) do
    template_text
    |> split_into_sentences()
    |> Enum.reduce_while({:ok, []}, fn sentence, {:ok, chunks} ->
      case classify_and_embed(world_id, sentence) do
        {:ok, chunk} -> {:cont, {:ok, [chunk | chunks]}}
        {:error, :vocabulary_building} = building -> {:halt, building}
      end
    end)
    |> case do
      {:ok, chunks} -> {:ok, Enum.reverse(chunks)}
      building -> building
    end
  end

  @doc "`segment/3` that does not wait for a rebuild, as `try_segment/2`.\n"
  @spec try_segment(String.t(), String.t(), term()) :: {:ok, [%Chunk{}]} | {:error, :vocabulary_building}
  def try_segment(world_id, template_text, source_intent)
      when is_binary(world_id) and is_binary(template_text) do
    with {:ok, chunks} <- try_segment(world_id, template_text) do
      {:ok, Enum.map(chunks, fn chunk -> %{chunk | source_intent: source_intent} end)}
    end
  end

  @doc "`segment_all/2` that does not wait for a rebuild, as `try_segment/2`.\n"
  @spec try_segment_all(String.t(), map()) :: {:ok, [%Chunk{}]} | {:error, :vocabulary_building}
  def try_segment_all(world_id, templates_by_intent) when is_binary(world_id) and is_map(templates_by_intent) do
    templates_by_intent
    |> Enum.flat_map(fn {intent, templates} -> Enum.map(templates, &{intent, &1}) end)
    |> Enum.reduce_while({:ok, []}, fn {intent, template}, {:ok, acc} ->
      case try_segment(world_id, template, intent) do
        {:ok, chunks} -> {:cont, {:ok, [chunks | acc]}}
        {:error, :vocabulary_building} = building -> {:halt, building}
      end
    end)
    |> case do
      {:ok, chunk_lists} -> {:ok, chunk_lists |> Enum.reverse() |> List.flatten()}
      building -> building
    end
  end

  # Runs `attempt` until the world's vocabulary is not being rebuilt, waiting
  # for each rebuild to finish.
  defp awaiting_rebuild(world_id, attempt) do
    case attempt.() do
      {:ok, result} ->
        result

      {:error, :vocabulary_building} ->
        :ok = WorldEmbedder.await_ready(world_id)
        awaiting_rebuild(world_id, attempt)
    end
  end

  @doc """
  Segments a template text into typed chunks, preserving $placeholder slot tokens.

  Extracts slot names and enrichment fields from $placeholder patterns before
  splitting, then attaches them to each resulting chunk. Chunks are embedded in
  the world's embedder, waiting for a rebuild of its vocabulary to finish as
  `segment/2` does.
  """
  def segment_with_slots(world_id, template_text) when is_binary(world_id) and is_binary(template_text) do
    {slots, enrichment_fields} = extract_slot_info(template_text)

    world_id
    |> segment(template_text)
    |> Enum.map(fn chunk -> %{chunk | slots: slots, enrichment_fields: enrichment_fields} end)
  end

  @doc """
  Segments a template with slots in the world's embedder and associates it with
  an intent.
  """
  def segment_with_slots(world_id, template_text, source_intent)
      when is_binary(world_id) and is_binary(template_text) do
    world_id
    |> segment_with_slots(template_text)
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

    tone_vector = [pos, neg, neu, conf, polarity, arousal, warmth, formality, playfulness, directness]

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

  @doc """
  Forgets the world's chunk type centroids, for a world whose vocabulary was
  rebuilt or that is destroyed (`Brain.EmbeddingCaches.forget_world/1`); the
  world's next classification builds them with the world's vocabulary.
  """
  @spec forget_world(String.t()) :: :ok
  def forget_world(world_id) when is_binary(world_id) do
    :persistent_term.erase(centroid_key(world_id))
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

  defp classify_and_embed(world_id, sentence) do
    with {:ok, embedding} <- world_embedding(world_id, sentence),
         {:ok, chunk_type} <- chunk_type(world_id, sentence, embedding) do
      {:ok, %Chunk{text: sentence, type: chunk_type, embedding: embedding, source_intent: nil}}
    end
  end

  defp chunk_type(world_id, sentence, embedding) do
    case classify_by_heuristic(sentence) do
      :body -> classify_chunk_type(world_id, embedding)
      heuristic_type -> {:ok, heuristic_type}
    end
  end

  defp classify_chunk_type(world_id, embedding) do
    with {:ok, centroids} <- get_or_build_centroids(world_id) do
      {best_type, _best_similarity} =
        centroids
        |> Enum.map(fn {type, centroid} ->
          similarity = FourthWall.Math.cosine_similarity!(embedding, centroid)
          {type, similarity}
        end)
        |> Enum.max_by(fn {_type, sim} -> sim end, fn -> {:body, 0.0} end)

      {:ok, best_type}
    end
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
        (List.last(tokens) == "?" and length(tokens) < 8) ->
        :clarification

      (first_token == "i" and Enum.at(tokens, 1) == "understand") or
        (first_token == "got" and Enum.at(tokens, 1) == "it") or
        first_token in @acknowledgment_tokens ->
        :acknowledgment

      true ->
        :body
    end
  end

  # The world's chunk type centroids, built once per world and shared by every
  # process (`:persistent_term`) until the world's vocabulary is rebuilt or the
  # world is destroyed (`forget_world/1`). Centroids are not kept while the
  # world's vocabulary is being rebuilt, so the next call builds them from the
  # rebuilt vocabulary.
  defp get_or_build_centroids(world_id) do
    case :persistent_term.get(centroid_key(world_id), nil) do
      nil ->
        with {:ok, centroids} <- build_centroids(world_id) do
          :persistent_term.put(centroid_key(world_id), centroids)
          {:ok, centroids}
        end

      centroids ->
        {:ok, centroids}
    end
  end

  defp centroid_key(world_id), do: {__MODULE__, @centroid_key, world_id}

  defp build_centroids(world_id) do
    Enum.reduce_while(@chunk_type_seeds, {:ok, %{}}, fn {type, seeds}, {:ok, acc} ->
      embedded =
        Enum.reduce_while(seeds, {:ok, []}, fn seed, {:ok, embeddings} ->
          case world_embedding(world_id, seed) do
            {:ok, embedding} -> {:cont, {:ok, [embedding | embeddings]}}
            {:error, :vocabulary_building} = building -> {:halt, building}
          end
        end)

      case embedded do
        {:ok, []} -> {:cont, {:ok, acc}}
        {:ok, embeddings} -> {:cont, {:ok, Map.put(acc, type, average_vectors(Enum.reverse(embeddings)))}}
        {:error, :vocabulary_building} = building -> {:halt, building}
      end
    end)
  end

  # A world's embedding comes from the world's embedder and nowhere else. While
  # its vocabulary is being rebuilt this is `{:error, :vocabulary_building}`,
  # which the public functions wait out and the `try_` ones return. A world
  # whose embedder has no copy, no vocabulary, or no table raises.
  defp world_embedding(world_id, text) do
    case WorldEmbedder.embed(world_id, text) do
      {:ok, embedding} ->
        {:ok, embedding}

      {:error, :vocabulary_building} ->
        {:error, :vocabulary_building}

      {:error, reason} ->
        raise "ChunkSegmenter: world #{inspect(world_id)}'s embedder cannot embed: #{inspect(reason)}"
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
end
