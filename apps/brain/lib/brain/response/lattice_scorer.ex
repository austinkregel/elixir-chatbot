defmodule Brain.Response.LatticeScorer do
  @moduledoc """
  Scores phrase inventory fragments for the lattice realizer.

  Each fragment is scored by combining:
  1. Cosine similarity between input feature vector and fragment prototype vector
  2. Tone compatibility between desired tone and fragment tone vector
  3. Additive intent bonuses (exact `source_intent` or same domain prefix)

  Weights and bonuses are loaded from `priv/response/system_config.json` under `"lattice"`.
  """

  alias Brain.Analysis.IntentUtils
  alias Brain.Response.PhraseInventory.Fragment

  @tone_dim 10
  @valence_dims 5

  @weights_path "priv/ml_models/lattice/scorer_weights.json"
  @config_path "priv/response/system_config.json"

  @default_lattice_config %{
    "feature_weight" => 0.55,
    "tone_weight" => 0.45,
    "intent_match_boost" => 0.10,
    "intent_domain_bonus" => 0.05
  }

  @doc """
  Scores a list of fragments against the input feature vector and desired tone.

  Options:
    - `:weights` - per-dimension scorer weights list
    - `:intent` - classified intent label for intent bonuses
    - `:lattice_config` - override lattice scoring config map

  Returns `[{%Fragment{}, score}]` sorted by score descending.
  """
  def score_fragments(fragments, input_fv, desired_tone, opts \\ []) do
    weights = Keyword.get(opts, :weights) || load_weights_or_nil()
    intent = Keyword.get(opts, :intent)
    config = Keyword.get(opts, :lattice_config) || load_lattice_config()

    feature_weight = Map.get(config, "feature_weight", 0.55)
    tone_weight = Map.get(config, "tone_weight", 0.45)
    intent_match_boost = Map.get(config, "intent_match_boost", 0.10)
    intent_domain_bonus = Map.get(config, "intent_domain_bonus", 0.05)

    fragments
    |> Enum.map(fn %Fragment{} = frag ->
      fv_score = feature_similarity(input_fv, frag.prototype_vector, weights)
      tone_score = tone_compatibility(desired_tone, frag.tone_vector)
      base = fv_score * feature_weight + tone_score * tone_weight
      bonus = intent_bonus(frag, intent, intent_match_boost, intent_domain_bonus)
      {frag, base + bonus}
    end)
    |> Enum.sort_by(fn {_f, score} -> score end, :desc)
  end

  defp intent_bonus(_frag, intent, _match_boost, _domain_bonus)
       when not is_binary(intent) or intent == "",
       do: 0.0

  defp intent_bonus(%Fragment{source_intent: source}, intent, match_boost, domain_bonus)
       when is_binary(source) do
    cond do
      source == intent -> match_boost
      IntentUtils.same_domain_prefix?(source, intent) -> domain_bonus
      true -> 0.0
    end
  end

  defp intent_bonus(_, _, _, _), do: 0.0

  @doc """
  Loads lattice scoring config from system_config.json.
  """
  def load_lattice_config do
    path = brain_priv(@config_path)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"lattice" => lattice}} when is_map(lattice) ->
            Map.merge(@default_lattice_config, lattice)

          _ ->
            @default_lattice_config
        end

      {:error, _} ->
        @default_lattice_config
    end
  end

  @doc """
  Computes the desired tone vector by blending user sentiment with domain tone bias.

  - `input_sentiment` - 5-float list from feature vector sentiment dims [pos, neg, neu, conf, polarity]
  - `tone_bias_vector` - 10-float tone vector from system_config.json
  - `mirror_coefficient` - float 0.0-1.0 controlling blend (0 = all bias, 1 = all input)
  """
  def compute_desired_tone(input_sentiment, tone_bias_vector, mirror_coefficient)
      when is_list(input_sentiment) and is_list(tone_bias_vector) do
    input_padded =
      case length(input_sentiment) do
        n when n >= @valence_dims -> Enum.take(input_sentiment, @valence_dims)
        n -> input_sentiment ++ List.duplicate(0.0, @valence_dims - n)
      end

    bias_valence = Enum.take(tone_bias_vector, @valence_dims)
    bias_personality = Enum.drop(tone_bias_vector, @valence_dims)

    blended_valence =
      Enum.zip(input_padded, bias_valence)
      |> Enum.map(fn {inp, bias} ->
        mirror_coefficient * inp + (1.0 - mirror_coefficient) * bias
      end)

    personality =
      case length(bias_personality) do
        n when n >= @tone_dim - @valence_dims -> Enum.take(bias_personality, @tone_dim - @valence_dims)
        n -> bias_personality ++ List.duplicate(0.5, @tone_dim - @valence_dims - n)
      end

    blended_valence ++ personality
  end

  def compute_desired_tone(_, tone_bias_vector, _) when is_list(tone_bias_vector) do
    tone_bias_vector
  end

  def compute_desired_tone(_, _, _) do
    List.duplicate(0.5, @tone_dim)
  end

  @doc "Loads scorer weights from disk if available."
  def load_weights do
    case load_weights_or_nil() do
      nil -> {:error, :not_found}
      w -> {:ok, w}
    end
  end

  defp load_weights_or_nil do
    path = brain_priv(@weights_path)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"weights" => w}} when is_list(w) -> w
          _ -> nil
        end

      {:error, _} ->
        nil
    end
  end

  defp feature_similarity(input_fv, prototype_fv, weights)
       when is_list(input_fv) and is_list(prototype_fv) and input_fv != [] and prototype_fv != [] do
    {a, b} = equalize_lengths(input_fv, prototype_fv)

    {a, b} =
      if is_list(weights) and length(weights) == length(a) do
        wa = Enum.zip(a, weights) |> Enum.map(fn {v, w} -> v * w end)
        wb = Enum.zip(b, weights) |> Enum.map(fn {v, w} -> v * w end)
        {wa, wb}
      else
        {a, b}
      end

    cosine_similarity(a, b)
  end

  defp feature_similarity(_, _, _), do: 0.0

  defp tone_compatibility(desired, fragment_tone)
       when is_list(desired) and is_list(fragment_tone) do
    {a, b} = equalize_lengths(desired, fragment_tone)
    sim = cosine_similarity(a, b)
    max(0.0, sim)
  end

  defp tone_compatibility(_, _), do: 0.5

  defp equalize_lengths(a, b) do
    la = length(a)
    lb = length(b)

    cond do
      la == lb -> {a, b}
      la < lb -> {a ++ List.duplicate(0.0, lb - la), b}
      true -> {a, b ++ List.duplicate(0.0, la - lb)}
    end
  end

  defp cosine_similarity(a, b) do
    dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, sum -> sum + x * y end)
    mag_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, sum -> sum + x * x end))
    mag_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, sum -> sum + x * x end))

    if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end

  defp brain_priv(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} -> Path.join("apps/brain", relative)
      priv_dir -> Path.join(priv_dir, Path.relative_to(relative, "priv"))
    end
  end
end
