defmodule Brain.Analysis.ChunkProfile.Projections do
  @moduledoc """
  Pure projection functions called by `ChunkProfile.materialize/2`.

  Each function derives a single semantic axis from the combined analysis
  map produced by the pipeline. Keeping them here avoids bloating the
  struct module with logic.
  """

  alias Brain.ML.MicroClassifiers

  @first_person_tokens MapSet.new(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"])
  @second_person_tokens MapSet.new(["you", "your", "yours", "yourself", "yourselves"])
  @negation_tokens MapSet.new(["not", "n't", "never", "no", "neither", "nor"])

  # ---------------------------------------------------------------------------
  # project_target/1
  # ---------------------------------------------------------------------------

  @doc """
  Determines who the utterance is *about* by voting across discourse
  analysis, MicroClassifier signals, and POS-tag pronoun counts.

  Returns one of `:agent`, `:self`, `:other_person`, `:world`, or `:ambiguous`.
  """
  def project_target(analysis) when is_map(analysis) do
    votes =
      [
        vote_from_discourse(analysis),
        vote_from_directed_classifier(analysis),
        vote_from_personal_classifier(analysis),
        vote_from_pronouns(analysis)
      ]
      |> Enum.reject(&is_nil/1)

    majority_vote(votes, :ambiguous)
  end

  def project_target(_), do: :ambiguous

  defp vote_from_discourse(analysis) do
    case get_in_safe(analysis, [:discourse, :addressee]) do
      :bot -> :agent
      :user -> :self
      :third_party -> :other_person
      _ -> nil
    end
  end

  defp vote_from_directed_classifier(analysis) do
    text = extract_text(analysis)

    case classify_axis(:directed_at_bot, text, nil) do
      :directed -> :agent
      _ -> nil
    end
  end

  defp vote_from_personal_classifier(analysis) do
    text = extract_text(analysis)

    case classify_axis(:personal_question, text, nil) do
      :personal -> :self
      _ -> nil
    end
  end

  defp vote_from_pronouns(analysis) do
    pos_tags = extract_pos_tags(analysis)

    {first_count, second_count} =
      Enum.reduce(pos_tags, {0, 0}, fn tagged, {fc, sc} ->
        {token, tag} = normalize_tagged(tagged)
        lower = String.downcase(token)

        cond do
          tag == :PRON and MapSet.member?(@first_person_tokens, lower) -> {fc + 1, sc}
          tag == :PRON and MapSet.member?(@second_person_tokens, lower) -> {fc, sc + 1}
          true -> {fc, sc}
        end
      end)

    cond do
      first_count > second_count and first_count > 0 -> :self
      second_count > first_count and second_count > 0 -> :agent
      true -> nil
    end
  end

  # ---------------------------------------------------------------------------
  # project_modality/1
  # ---------------------------------------------------------------------------

  @doc """
  Derives the utterance modality from the speech-act classification.

  Returns `:interrogative`, `:imperative`, `:exclamatory`, or `:declarative`.
  """
  def project_modality(analysis) when is_map(analysis) do
    speech_act = Map.get(analysis, :speech_act) || %{}

    cond do
      Map.get(speech_act, :is_question) == true -> :interrogative
      Map.get(speech_act, :is_imperative) == true -> :imperative
      Map.get(speech_act, :category) == :expressive -> :exclamatory
      true -> :declarative
    end
  end

  def project_modality(_), do: :declarative

  # ---------------------------------------------------------------------------
  # project_polarity/1
  # ---------------------------------------------------------------------------

  @doc """
  Counts negation particles in the POS-tagged token stream and returns
  `:negative` when at least one is present, `:affirmative` otherwise.

  Membership is checked via MapSet — no regex or `String.contains?`.
  """
  def project_polarity(analysis) when is_map(analysis) do
    pos_tags = extract_pos_tags(analysis)

    negation_count =
      Enum.count(pos_tags, fn tagged ->
        {token, tag} = normalize_tagged(tagged)
        lower = String.downcase(token)
        tag == :PART and MapSet.member?(@negation_tokens, lower)
      end)

    if negation_count > 0, do: :negative, else: :affirmative
  end

  def project_polarity(_), do: :affirmative

  # ---------------------------------------------------------------------------
  # project_sentiment_alignment/1
  # ---------------------------------------------------------------------------

  @doc """
  Checks whether the sentiment polarity is congruent with the
  speech-act category.

  Returns `:congruent`, `:incongruent`, or `:neutral`.
  """
  def project_sentiment_alignment(analysis) when is_map(analysis) do
    sentiment_label = get_in_safe(analysis, [:sentiment, :label])
    speech_category = get_in_safe(analysis, [:speech_act, :category])

    cond do
      sentiment_label in [nil, :neutral] ->
        :neutral

      sentiment_label == :positive and speech_category in [:assertive, :expressive] ->
        :congruent

      sentiment_label == :negative and speech_category in [:directive, :commissive] ->
        :incongruent

      sentiment_label == :positive and speech_category in [:directive, :commissive] ->
        :congruent

      sentiment_label == :negative and speech_category in [:assertive, :expressive] ->
        :incongruent

      true ->
        :neutral
    end
  end

  def project_sentiment_alignment(_), do: :neutral

  # ---------------------------------------------------------------------------
  # project_response_posture/1
  # ---------------------------------------------------------------------------

  @doc """
  Given aggregated confidence signals, determines how the system should
  frame its reply.

  Expects a map with keys: `certainty`, `confidence`, `slot_completeness`,
  `novelty_score`.

  Returns `:direct`, `:clarify`, `:tentative_confirm`, or `:hedged`.
  """
  def project_response_posture(signals) when is_map(signals) do
    confidence = Map.get(signals, :confidence, 0.0)
    certainty = Map.get(signals, :certainty)
    slot_completeness = Map.get(signals, :slot_completeness, 0.0)
    novelty_score = Map.get(signals, :novelty_score, 0.0)

    cond do
      confidence >= 0.7 and certainty in [:committed, :tentative] and
          slot_completeness >= 0.8 and novelty_score < 0.5 ->
        :direct

      confidence < 0.4 or novelty_score >= 0.7 ->
        :clarify

      confidence >= 0.5 and certainty in [:hedged, :speculative] ->
        :tentative_confirm

      true ->
        :hedged
    end
  end

  def project_response_posture(_), do: :hedged

  # ---------------------------------------------------------------------------
  # project_engagement_level/1
  # ---------------------------------------------------------------------------

  @doc """
  Determines the user's engagement level from addressee, target, modality,
  and urgency.

  Returns `:urgent_demand`, `:active_request`, `:casual_engagement`, or
  `:passive_observation`.
  """
  def project_engagement_level(signals) when is_map(signals) do
    addressee = Map.get(signals, :addressee)
    target = Map.get(signals, :target)
    modality = Map.get(signals, :modality)
    urgency = Map.get(signals, :urgency)

    cond do
      addressee == :bot and modality == :imperative and urgency in [:high, :critical] ->
        :urgent_demand

      addressee == :bot and modality == :interrogative ->
        :active_request

      addressee == :bot and modality in [:imperative, :declarative] ->
        :casual_engagement

      addressee in [:user, :third_party] or target == :world ->
        :passive_observation

      true ->
        :casual_engagement
    end
  end

  def project_engagement_level(_), do: :casual_engagement

  # ---------------------------------------------------------------------------
  # project_self_disclosure_level/1
  # ---------------------------------------------------------------------------

  @doc """
  Measures how much personal information the user is volunteering.

  Expects a map with keys: `target`, `domain`, `certainty`, `sentiment`.

  Returns `:none`, `:factual_self_info`, `:emotional_self_disclosure`,
  `:opinion`, or `:preference`.
  """
  def project_self_disclosure_level(signals) when is_map(signals) do
    target = Map.get(signals, :target)
    domain = Map.get(signals, :domain)
    certainty = Map.get(signals, :certainty)
    sentiment_label = get_in_safe(signals, [:sentiment, :label])

    cond do
      target != :self ->
        :none

      domain in [:introduction, :smalltalk] and certainty == :committed ->
        :factual_self_info

      sentiment_label in [:positive, :negative] and certainty in [:committed, :tentative] ->
        :emotional_self_disclosure

      certainty == :speculative ->
        :opinion

      certainty == :hedged ->
        :preference

      true ->
        :factual_self_info
    end
  end

  def project_self_disclosure_level(_), do: :none

  # ---------------------------------------------------------------------------
  # project_temporal_framing/1
  # ---------------------------------------------------------------------------

  @doc """
  Combines tense, aspect, and polarity into a temporal-frame label.

  Expects a map with keys: `tense`, `aspect`, `polarity`.

  Returns `:negated_past`, `:completed_past`, `:ongoing`,
  `:hypothetical_future`, or `:timeless`.
  """
  def project_temporal_framing(signals) when is_map(signals) do
    tense = Map.get(signals, :tense)
    aspect = Map.get(signals, :aspect)
    polarity = Map.get(signals, :polarity)

    cond do
      tense == :past and polarity == :negative ->
        :negated_past

      tense == :past and aspect == :simple ->
        :completed_past

      tense == :present and aspect in [:progressive, :perfect_progressive] ->
        :ongoing

      tense == :future ->
        :hypothetical_future

      tense == :atemporal ->
        :timeless

      true ->
        :timeless
    end
  end

  def project_temporal_framing(_), do: :timeless

  # ---------------------------------------------------------------------------
  # classify_axis/3
  # ---------------------------------------------------------------------------

  @doc """
  Convenience wrapper around `MicroClassifiers.classify/2`.

  Converts the returned string label to an atom and falls back to
  `default` on any error or when the server is not loaded.
  """
  def classify_axis(classifier_name, text, default) when is_atom(classifier_name) do
    try do
      case MicroClassifiers.classify(classifier_name, text) do
        {:ok, label, _score} when is_binary(label) -> safe_to_atom(label)
        _ -> default
      end
    rescue
      _ -> default
    end
  end

  def classify_axis(_classifier_name, _text, default), do: default

  # ---------------------------------------------------------------------------
  # safe_to_atom/1
  # ---------------------------------------------------------------------------

  @doc """
  Converts a binary to an atom, preferring `String.to_existing_atom/1`
  to avoid atom-table pollution. Falls back to `String.to_atom/1` for
  genuinely new labels.
  """
  def safe_to_atom(value) when is_binary(value) do
    try do
      String.to_existing_atom(value)
    rescue
      ArgumentError -> String.to_atom(value)
    end
  end

  def safe_to_atom(value) when is_atom(value), do: value
  def safe_to_atom(_), do: :unknown

  # ---------------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------------

  defp extract_text(%{text: text}) when is_binary(text), do: text
  defp extract_text(%{raw_input: text}) when is_binary(text), do: text
  defp extract_text(_), do: ""

  defp extract_pos_tags(%{pos_tags: tags}) when is_list(tags), do: tags
  defp extract_pos_tags(_), do: []

  defp normalize_tagged({token, tag}) when is_binary(token), do: {token, tag}
  defp normalize_tagged(tag) when is_atom(tag), do: {"", tag}
  defp normalize_tagged(_), do: {"", :X}

  defp get_in_safe(map, keys) when is_map(map) do
    Enum.reduce_while(keys, map, fn key, acc ->
      case acc do
        %{^key => val} -> {:cont, val}
        _ -> {:halt, nil}
      end
    end)
  end

  defp majority_vote([], default), do: default

  defp majority_vote(votes, default) do
    frequencies = Enum.frequencies(votes)

    {winner, max_count} = Enum.max_by(frequencies, fn {_k, v} -> v end)

    tied =
      frequencies
      |> Enum.filter(fn {_k, v} -> v == max_count end)
      |> length()

    if tied > 1, do: default, else: winner
  end
end
