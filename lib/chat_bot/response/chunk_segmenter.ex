defmodule ChatBot.Response.ChunkSegmenter do
  @moduledoc """
  Segments response templates into typed chunks for template blending.

  Uses sentence boundaries and embedding-based classification to identify
  chunk types. Chunks can then be recombined to generate novel responses.

  ## Chunk Types

  - `:greeting` - Opening phrases ("Hello!", "Nice to meet you!")
  - `:acknowledgment` - Confirmation phrases ("I understand.", "Got it.")
  - `:body` - Substantive content ("The weather in $location is...")
  - `:offer` - Invitations for next action ("What can I help with?")
  - `:clarification` - Requests for missing info ("Which location?")
  - `:closing` - Farewell phrases ("Have a great day!", "Goodbye!")

  ## Usage

      ChunkSegmenter.segment("Hello! The weather is sunny. Anything else?")
      # => [
      #   %Chunk{text: "Hello!", type: :greeting},
      #   %Chunk{text: "The weather is sunny.", type: :body},
      #   %Chunk{text: "Anything else?", type: :offer}
      # ]
  """

  alias ChatBot.Memory.Embedder

  require Logger

  # Chunk struct
  defmodule Chunk do
    @moduledoc "A segmented piece of a response template"
    defstruct [:text, :type, :embedding, :source_intent]
  end

  # Seed examples for each chunk type (used to build centroids)
  @chunk_type_seeds %{
    greeting: [
      "Hello!",
      "Hi there!",
      "Hey!",
      "Good morning!",
      "Nice to meet you!",
      "Welcome!",
      "Greetings!"
    ],
    acknowledgment: [
      "I understand.",
      "Got it.",
      "Okay.",
      "I see.",
      "Understood.",
      "Right.",
      "Sure thing."
    ],
    body: [
      "The weather is sunny.",
      "Playing your music now.",
      "Here's what I found.",
      "The temperature is 72 degrees.",
      "I'll set that reminder for you."
    ],
    offer: [
      "What can I help you with?",
      "Anything else?",
      "How can I assist you?",
      "What would you like to do?",
      "Is there something else you need?"
    ],
    clarification: [
      "Which location did you mean?",
      "What time would you like?",
      "I need to know the date.",
      "Could you specify?",
      "Which one?"
    ],
    closing: [
      "Have a great day!",
      "Goodbye!",
      "Talk to you later!",
      "Take care!",
      "See you soon!"
    ]
  }

  # Cached centroids (built lazily)
  @centroid_key :chunk_type_centroids

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Segments a template text into typed chunks.

  Returns a list of %Chunk{} structs, each with:
  - text: The chunk text
  - type: The classified chunk type (:greeting, :body, etc.)
  - embedding: TF-IDF embedding for the chunk
  """
  def segment(template_text) when is_binary(template_text) do
    template_text
    |> split_into_sentences()
    |> Enum.map(&classify_and_embed/1)
    |> Enum.filter(& &1)
  end

  @doc """
  Segments a template and associates it with an intent.
  """
  def segment(template_text, source_intent) when is_binary(template_text) do
    template_text
    |> segment()
    |> Enum.map(fn chunk -> %{chunk | source_intent: source_intent} end)
  end

  @doc """
  Segments all templates from a map of {intent => [template_texts]}.
  Returns a flat list of all chunks.
  """
  def segment_all(templates_by_intent) when is_map(templates_by_intent) do
    templates_by_intent
    |> Enum.flat_map(fn {intent, templates} ->
      Enum.flat_map(templates, fn template ->
        segment(template, intent)
      end)
    end)
  end

  @doc """
  Returns the chunk type seeds used for classification.
  """
  def get_type_seeds, do: @chunk_type_seeds

  @doc """
  Clears the cached centroids (useful for testing).
  """
  def clear_centroids do
    Process.delete(@centroid_key)
    :ok
  end

  # ============================================================================
  # Sentence Splitting
  # ============================================================================

  @doc """
  Splits text into sentences using punctuation boundaries.
  """
  def split_into_sentences(text) when is_binary(text) do
    # Use tokenizer if available, otherwise simple split
    text
    |> String.split(~r/(?<=[.!?])\s+/, trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.filter(&(String.length(&1) > 0))
  end

  # ============================================================================
  # Classification
  # ============================================================================

  defp classify_and_embed(sentence) do
    case Embedder.embed(sentence) do
      {:ok, embedding} ->
        chunk_type = classify_chunk_type(embedding)

        %Chunk{
          text: sentence,
          type: chunk_type,
          embedding: embedding,
          source_intent: nil
        }

      _ ->
        # Fallback: use heuristic classification
        chunk_type = classify_by_heuristic(sentence)

        %Chunk{
          text: sentence,
          type: chunk_type,
          embedding: nil,
          source_intent: nil
        }
    end
  end

  defp classify_chunk_type(embedding) do
    centroids = get_or_build_centroids()

    # Find nearest centroid
    {best_type, _best_similarity} =
      centroids
      |> Enum.map(fn {type, centroid} ->
        similarity = cosine_similarity(embedding, centroid)
        {type, similarity}
      end)
      |> Enum.max_by(fn {_type, sim} -> sim end, fn -> {:body, 0.0} end)

    best_type
  end

  defp classify_by_heuristic(sentence) do
    lower = String.downcase(sentence)

    cond do
      # Greeting patterns
      String.starts_with?(lower, "hello") or
      String.starts_with?(lower, "hi") or
      String.starts_with?(lower, "hey") or
      String.starts_with?(lower, "good morning") or
      String.starts_with?(lower, "good afternoon") or
      String.starts_with?(lower, "nice to meet") or
        String.starts_with?(lower, "welcome") ->
        :greeting

      # Closing patterns
      String.starts_with?(lower, "goodbye") or
      String.starts_with?(lower, "bye") or
      String.starts_with?(lower, "take care") or
      String.starts_with?(lower, "see you") or
        String.contains?(lower, "have a great day") ->
        :closing

      # Offer patterns (questions offering help)
      String.contains?(lower, "can i help") or
      String.contains?(lower, "anything else") or
      String.contains?(lower, "what would you like") or
        String.ends_with?(lower, "?") and String.contains?(lower, "help") ->
        :offer

      # Clarification patterns (questions asking for info)
      String.starts_with?(lower, "which") or
      String.starts_with?(lower, "what") or
      String.contains?(lower, "need to know") or
        (String.ends_with?(lower, "?") and String.length(sentence) < 30) ->
        :clarification

      # Acknowledgment patterns
      String.starts_with?(lower, "i understand") or
      String.starts_with?(lower, "got it") or
      String.starts_with?(lower, "okay") or
      String.starts_with?(lower, "sure") or
        lower == "right." ->
        :acknowledgment

      # Default to body
      true ->
        :body
    end
  end

  # ============================================================================
  # Centroids
  # ============================================================================

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
          if length(embeddings) > 0 do
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

  defp average_vectors(vectors) when is_list(vectors) and length(vectors) > 0 do
    n = length(vectors)
    vec_length = length(List.first(vectors))

    # Sum all vectors element-wise
    summed =
      Enum.reduce(vectors, List.duplicate(0.0, vec_length), fn vec, acc ->
        Enum.zip(vec, acc)
        |> Enum.map(fn {a, b} -> a + b end)
      end)

    # Divide by count to get average
    Enum.map(summed, fn x -> x / n end)
  end

  defp average_vectors(_), do: nil

  # ============================================================================
  # Similarity
  # ============================================================================

  defp cosine_similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    if length(vec1) != length(vec2) do
      0.0
    else
      dot = Enum.zip(vec1, vec2) |> Enum.reduce(0.0, fn {a, b}, sum -> sum + a * b end)
      mag1 = :math.sqrt(Enum.reduce(vec1, 0.0, fn x, sum -> sum + x * x end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0.0, fn x, sum -> sum + x * x end))

      if mag1 == 0.0 or mag2 == 0.0, do: 0.0, else: dot / (mag1 * mag2)
    end
  end

  defp cosine_similarity(_, _), do: 0.0
end
