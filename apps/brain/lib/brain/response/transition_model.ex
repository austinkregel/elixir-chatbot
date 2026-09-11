defmodule Brain.Response.TransitionModel do
  @moduledoc """
  Scores the smoothness of transitions between adjacent phrase fragments
  in the lattice realizer.

  Uses token bigram statistics at fragment boundaries to evaluate how
  naturally two fragments join together.
  """

  alias Brain.ML.Tokenizer

  @scores_path "priv/ml_models/lattice/transition_scores.json"
  @boundary_window 3

  @doc """
  Scores the transition from fragment_a to fragment_b.

  Examines the last @boundary_window tokens of fragment_a and the first
  @boundary_window tokens of fragment_b, computing a smoothness score
  based on bigram transition statistics.

  Returns a float between 0.0 and 1.0.
  """
  def score(fragment_a_text, fragment_b_text, opts \\ []) do
    bigram_scores = Keyword.get(opts, :bigram_scores, load_bigram_scores())

    tokens_a = extract_boundary_tokens(fragment_a_text, :tail)
    tokens_b = extract_boundary_tokens(fragment_b_text, :head)

    if tokens_a == [] or tokens_b == [] do
      0.5
    else
      boundary_bigrams = build_boundary_bigrams(tokens_a, tokens_b)

      if boundary_bigrams == [] do
        0.5
      else
        scores =
          Enum.map(boundary_bigrams, fn bigram_key ->
            Map.get(bigram_scores, bigram_key, 0.3)
          end)

        Enum.sum(scores) / length(scores)
      end
    end
  end

  @doc """
  Scores a full path of fragment texts, returning the average transition score.
  """
  def score_path(fragment_texts, opts \\ [])
  def score_path([], _opts), do: 1.0
  def score_path([_single], _opts), do: 1.0

  def score_path(fragment_texts, opts) when is_list(fragment_texts) do
    bigram_scores = Keyword.get(opts, :bigram_scores, load_bigram_scores())
    opts_with_scores = Keyword.put(opts, :bigram_scores, bigram_scores)

    pairs = Enum.zip(fragment_texts, tl(fragment_texts))

    scores =
      Enum.map(pairs, fn {a, b} ->
        score(a, b, opts_with_scores)
      end)

    Enum.sum(scores) / length(scores)
  end

  @doc """
  Builds bigram transition scores from a corpus of fragment sequences.

  Takes a list of fragment text sequences (each sequence is a list of
  consecutive fragment texts as they appear in training data).
  Returns a map of bigram_key => score.
  """
  def build_scores(fragment_sequences) when is_list(fragment_sequences) do
    bigram_counts =
      fragment_sequences
      |> Enum.flat_map(fn sequence ->
        pairs = Enum.zip(sequence, tl(sequence))

        Enum.flat_map(pairs, fn {a, b} ->
          tokens_a = extract_boundary_tokens(a, :tail)
          tokens_b = extract_boundary_tokens(b, :head)
          build_boundary_bigrams(tokens_a, tokens_b)
        end)
      end)
      |> Enum.frequencies()

    max_count = Enum.max(Map.values(bigram_counts), fn -> 1 end)

    Map.new(bigram_counts, fn {key, count} ->
      {key, count / max_count}
    end)
  end

  @doc "Saves bigram scores to disk."
  def save_scores(scores) when is_map(scores) do
    path = brain_priv(@scores_path)
    File.mkdir_p!(Path.dirname(path))

    data = %{
      "version" => 1,
      "generated_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "bigrams" => scores
    }

    File.write!(path, Jason.encode!(data, pretty: true))
    {:ok, path}
  end

  @doc "Loads bigram scores from disk."
  def load_bigram_scores do
    path = brain_priv(@scores_path)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"bigrams" => bigrams}} when is_map(bigrams) -> bigrams
          _ -> %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp extract_boundary_tokens(text, direction) when is_binary(text) do
    tokens =
      text
      |> Tokenizer.tokenize()
      |> Enum.map(fn t -> t.normalized || t.text end)
      |> Enum.filter(&(String.length(&1) > 0))

    case direction do
      :tail -> Enum.take(tokens, -@boundary_window)
      :head -> Enum.take(tokens, @boundary_window)
    end
  end

  defp extract_boundary_tokens(_, _), do: []

  defp build_boundary_bigrams(tail_tokens, head_tokens) do
    last = List.last(tail_tokens)
    first = List.first(head_tokens)

    boundary = if last && first, do: [bigram_key(last, first)], else: []

    within_tail =
      if match?([_, _ | _], tail_tokens) do
        tail_tokens
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] -> bigram_key(a, b) end)
      else
        []
      end

    within_head =
      if match?([_, _ | _], head_tokens) do
        head_tokens
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] -> bigram_key(a, b) end)
      else
        []
      end

    boundary ++ within_tail ++ within_head
  end

  defp bigram_key(a, b), do: "#{String.downcase(a)}|#{String.downcase(b)}"

  defp brain_priv(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} -> Path.join("apps/brain", relative)
      priv_dir -> Path.join(priv_dir, Path.relative_to(relative, "priv"))
    end
  end
end
