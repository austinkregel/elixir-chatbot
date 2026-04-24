defmodule Brain.Analysis.FeatureExtractor.WordFeatures do
  @moduledoc """
  Computes per-word feature vectors from POS tags and lexicon lookups.

  Each content word (NOUN, PROPN, VERB, ADJ, ADV) gets a feature vector containing:
  - POS tag (one-hot, 16 dims)
  - Lexical domain (one-hot, ~25 dims from WordNet lexicographer files)
  - Hypernym depth (1 dim, normalized)
  - Polysemy count (1 dim, normalized)
  - Is OOV flag (1 dim)
  - Position in chunk (1 dim, normalized 0-1)
  - Word length (1 dim, normalized)

  Total: ~46 dims per word.
  """

  alias Brain.Lexicon

  @all_pos [:NOUN, :PROPN, :VERB, :AUX, :ADJ, :ADV, :PRON, :DET, :ADP, :CONJ, :PART, :NUM, :INTJ, :PUNCT, :SYM, :X]

  @max_hypernym_depth 20.0
  @max_polysemy 30.0
  @max_word_length 20.0

  @doc """
  Extracts per-word features for all tokens in a chunk.

  Takes a list of `{token, pos_tag}` tuples and returns a list of
  word feature maps, one per token.

  ## Returns
  A list of maps, each containing:
  - `:token` - the original token
  - `:pos` - the POS tag
  - `:is_content_word` - whether this is a content word
  - `:features` - list of floats (the feature vector)
  - `:lexical_domain` - the resolved lexical domain atom (or nil)
  - `:is_oov` - whether the word is out of vocabulary
  - `:hypernym_depth` - raw hypernym depth
  - `:polysemy_count` - raw polysemy count
  """
  def extract(pos_tagged_tokens) when is_list(pos_tagged_tokens) do
    total_tokens = max(length(pos_tagged_tokens), 1)
    context_words = extract_context_words(pos_tagged_tokens)

    pos_tagged_tokens
    |> Enum.with_index()
    |> Enum.map(fn {{token, pos}, idx} ->
      position = idx / max(total_tokens - 1, 1)
      extract_word(token, pos, position, context_words)
    end)
  end

  @doc """
  Extracts features for content words only, returning a filtered list.
  """
  def extract_content_words(pos_tagged_tokens) when is_list(pos_tagged_tokens) do
    pos_tagged_tokens
    |> extract()
    |> Enum.filter(& &1.is_content_word)
  end

  @doc """
  Returns the dimension count for per-word feature vectors.
  """
  def vector_dimension do
    length(@all_pos) + length(Lexicon.domain_atoms()) + 4
  end

  # -- Private ----------------------------------------------------------------

  defp extract_word(token, pos, position, context_words) do
    normalized = String.downcase(token)
    is_content = content_word?(pos)

    {domain, is_oov, hyp_depth, poly_count} =
      if is_content do
        lookup_lexicon(normalized, pos, context_words)
      else
        {nil, false, 0, 0}
      end

    features = build_feature_vector(pos, domain, hyp_depth, poly_count, is_oov, position, token)

    %{
      token: token,
      pos: pos,
      is_content_word: is_content,
      features: features,
      lexical_domain: domain,
      is_oov: is_oov,
      hypernym_depth: hyp_depth,
      polysemy_count: poly_count
    }
  end

  defp lookup_lexicon(word, pos, context_words) do
    wn_pos = pos_to_wordnet(pos)

    if Lexicon.oov?(word) do
      {nil, true, 0, 0}
    else
      domain =
        case Lexicon.disambiguate(word, wn_pos, context_words) do
          {:ok, %{domain: d}} -> d
          _ -> Lexicon.primary_domain(word, wn_pos)
        end

      hyp_depth = Lexicon.hypernym_depth(word, wn_pos)
      poly_count = Lexicon.polysemy_count(word)

      {domain, false, hyp_depth, poly_count}
    end
  rescue
    _ -> {nil, true, 0, 0}
  end

  defp build_feature_vector(pos, domain, hyp_depth, poly_count, is_oov, _position, token) do
    pos_one_hot = one_hot_pos(pos)
    domain_one_hot = one_hot_domain(domain)

    hyp_norm = min(hyp_depth / @max_hypernym_depth, 1.0)
    poly_norm = min(poly_count / @max_polysemy, 1.0)
    oov_flag = if is_oov, do: 1.0, else: 0.0
    word_len = min(String.length(token) / @max_word_length, 1.0)

    pos_one_hot ++ domain_one_hot ++ [hyp_norm, poly_norm, oov_flag, word_len]
  end

  defp one_hot_pos(pos) do
    normalized = normalize_pos(pos)
    Enum.map(@all_pos, fn p -> if p == normalized, do: 1.0, else: 0.0 end)
  end

  defp one_hot_domain(nil) do
    Enum.map(Lexicon.domain_atoms(), fn _ -> 0.0 end)
  end

  defp one_hot_domain(domain) do
    Enum.map(Lexicon.domain_atoms(), fn d -> if d == domain, do: 1.0, else: 0.0 end)
  end

  defp normalize_pos(pos) when pos in [:noun, :NOUN, "NOUN", "noun"], do: :NOUN
  defp normalize_pos(pos) when pos in [:propn, :PROPN, "PROPN", "propn"], do: :PROPN
  defp normalize_pos(pos) when pos in [:verb, :VERB, "VERB", "verb"], do: :VERB
  defp normalize_pos(pos) when pos in [:adj, :ADJ, :adj_satellite, "ADJ", "adj"], do: :ADJ
  defp normalize_pos(pos) when pos in [:adv, :ADV, "ADV", "adv"], do: :ADV
  defp normalize_pos(pos) when pos in [:aux, :AUX, "AUX", "aux"], do: :AUX
  defp normalize_pos(pos) when pos in [:pron, :PRON, "PRON", "pron"], do: :PRON
  defp normalize_pos(pos) when pos in [:det, :DET, "DET", "det"], do: :DET
  defp normalize_pos(pos) when pos in [:adp, :ADP, "ADP", "adp"], do: :ADP
  defp normalize_pos(pos) when pos in [:conj, :CONJ, "CONJ", "conj", "CCONJ", "SCONJ"], do: :CONJ
  defp normalize_pos(pos) when pos in [:part, :PART, "PART", "part"], do: :PART
  defp normalize_pos(pos) when pos in [:num, :NUM, "NUM", "num"], do: :NUM
  defp normalize_pos(pos) when pos in [:intj, :INTJ, "INTJ", "intj"], do: :INTJ
  defp normalize_pos(pos) when pos in [:punct, :PUNCT, "PUNCT", "punct"], do: :PUNCT
  defp normalize_pos(pos) when pos in [:sym, :SYM, "SYM", "sym"], do: :SYM
  defp normalize_pos(_), do: :X

  defp content_word?(pos) do
    normalize_pos(pos) in [:NOUN, :PROPN, :VERB, :ADJ, :ADV]
  end

  defp pos_to_wordnet(pos) do
    case normalize_pos(pos) do
      p when p in [:NOUN, :PROPN] -> :noun
      :VERB -> :verb
      :ADJ -> :adj
      :ADV -> :adv
      _ -> nil
    end
  end

  defp extract_context_words(pos_tagged_tokens) do
    pos_tagged_tokens
    |> Enum.filter(fn {_token, pos} -> content_word?(pos) end)
    |> Enum.map(fn {token, _pos} -> String.downcase(token) end)
  end
end
