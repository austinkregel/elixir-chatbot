defmodule Brain.Response.LatticeScorerTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Response.LatticeScorer
  alias Brain.Response.PhraseInventory.Fragment

  @dim ChunkFeatures.vector_dimension()

  defp unit_vector(index) when index >= 0 and index < @dim do
    List.duplicate(0.0, @dim) |> List.replace_at(index, 1.0)
  end

  # Unit vector with cosine ~0.94 to unit_vector(0).
  defp near_input_vector do
    raw = [1.0, 0.35 | List.duplicate(0.0, @dim - 2)]
    normalize(raw)
  end

  # Unit vector with cosine ~0.86 to unit_vector(0).
  defp moderate_input_vector do
    raw = [1.0, 0.0, 0.6 | List.duplicate(0.0, @dim - 2)]
    normalize(raw)
  end

  defp normalize(raw) do
    norm = :math.sqrt(Enum.reduce(raw, 0.0, fn x, acc -> acc + x * x end))
    Enum.map(raw, fn x -> x / norm end)
  end

  defp frag(opts) do
    struct!(
      Fragment,
      Keyword.merge(
        [
          text: "test",
          chunk_type: "body",
          primitive_type: "content",
          primitive_variant: "enriched",
          tone_vector: List.duplicate(0.5, 10),
          prototype_vector: unit_vector(0),
          source_intent: "weather.query"
        ],
        opts
      )
    )
  end

  describe "score_fragments/4" do
    test "ranks fragment with closer prototype vector higher" do
      input = unit_vector(0)
      close = frag(prototype_vector: unit_vector(0), source_intent: "other.intent")
      far = frag(prototype_vector: unit_vector(5), source_intent: "other.intent")

      scored =
        LatticeScorer.score_fragments(
          [far, close],
          input,
          List.duplicate(0.5, 10),
          lattice_config: %{
            "feature_weight" => 1.0,
            "tone_weight" => 0.0,
            "intent_match_boost" => 0.0,
            "intent_domain_bonus" => 0.0
          }
        )

      assert [{^close, _}, {^far, _}] = scored
      {_, close_score} = hd(scored)
      {_, far_score} = List.last(scored)
      assert close_score > far_score
    end

    test "intent match bonus beats slightly closer prototype without match" do
      input = unit_vector(0)

      matched =
        frag(
          prototype_vector: moderate_input_vector(),
          source_intent: "weather.query"
        )

      closer_wrong_intent =
        frag(
          prototype_vector: near_input_vector(),
          source_intent: "smalltalk.greet"
        )

      scored =
        LatticeScorer.score_fragments(
          [closer_wrong_intent, matched],
          input,
          List.duplicate(0.5, 10),
          intent: "weather.query",
          lattice_config: %{
            "feature_weight" => 1.0,
            "tone_weight" => 0.0,
            "intent_match_boost" => 0.10,
            "intent_domain_bonus" => 0.0
          }
        )

      assert [{^matched, matched_score}, {^closer_wrong_intent, closer_score}] = scored
      assert matched_score > closer_score
    end

    test "domain bonus is less than exact intent match boost" do
      input = unit_vector(0)

      domain_match =
        frag(prototype_vector: unit_vector(8), source_intent: "weather.condition")

      exact_other =
        frag(prototype_vector: unit_vector(8), source_intent: "smalltalk.greet")

      config = %{
        "feature_weight" => 0.0,
        "tone_weight" => 0.0,
        "intent_match_boost" => 0.20,
        "intent_domain_bonus" => 0.05
      }

      [{_, domain_score}] =
        LatticeScorer.score_fragments([domain_match], input, [], intent: "weather.query", lattice_config: config)

      [{_, exact_score}] =
        LatticeScorer.score_fragments([exact_other], input, [], intent: "smalltalk.greet", lattice_config: config)

      assert exact_score > domain_score
    end
  end

  describe "load_lattice_config/0" do
    test "returns defaults when lattice block present in system_config" do
      config = LatticeScorer.load_lattice_config()
      assert Map.get(config, "feature_weight") == 0.55
      assert Map.get(config, "intent_match_boost") == 0.10
    end
  end
end
