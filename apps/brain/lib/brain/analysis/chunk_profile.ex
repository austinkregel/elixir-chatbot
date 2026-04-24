defmodule Brain.Analysis.ChunkProfile do
  @moduledoc """
  Materializes a rich profile from a feature vector and existing analysis signals.

  A ChunkProfile complements `Brain.Analysis.ChunkAnalysis` by projecting its
  raw signals into **17 primary axes** and **4 interaction axes**, plus a
  deterministic `derived_label`.

  Primary axes capture *what* the utterance is about (domain, speech act, tense,
  modality, polarity, etc.). Interaction axes capture *how* the system should
  engage (response posture, engagement level, self-disclosure, temporal framing).

  Axes are filled from three sources:

  1. **Hard copies** from existing analysis fields (speech act, discourse, slots)
  2. **MicroClassifier projections** for axes that need trained classification
     (domain, tense, aspect, urgency, certainty)
  3. **Compositional derivations** from multiple signals (target, modality,
     polarity, sentiment alignment, interaction axes)

  All MicroClassifier calls are wrapped in safe helpers that fall back to
  defaults when classifiers are not loaded.
  """

  alias Brain.Analysis.ChunkAnalysis
  alias Brain.ML.MicroClassifiers

  @type t :: %__MODULE__{
          domain: atom(),
          speech_act_category: atom(),
          speech_act_subtype: atom(),
          target: atom(),
          modality: atom(),
          polarity: atom(),
          tense: atom(),
          aspect: atom(),
          addressee: atom(),
          urgency: atom(),
          certainty: atom(),
          sentiment_alignment: atom(),
          slot_completeness: float(),
          novelty_score: float(),
          feature_provenance: map(),
          confidence: float(),
          derived_label: String.t(),
          response_posture: atom(),
          engagement_level: atom(),
          self_disclosure_level: atom(),
          temporal_framing: atom(),
          feature_vector: list(float())
        }

  defstruct domain: :unknown,
            speech_act_category: :unknown,
            speech_act_subtype: :unknown,
            target: :ambiguous,
            modality: :declarative,
            polarity: :affirmative,
            tense: :present,
            aspect: :simple,
            addressee: :unknown,
            urgency: :low,
            certainty: :committed,
            sentiment_alignment: :neutral,
            slot_completeness: 1.0,
            novelty_score: 0.0,
            feature_provenance: %{},
            confidence: 0.0,
            derived_label: "",
            response_posture: :direct,
            engagement_level: :casual_engagement,
            self_disclosure_level: :none,
            temporal_framing: :timeless,
            feature_vector: []

  # -------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------

  @doc """
  Creates a default profile with all axes at their zero-state defaults.
  """
  @spec new() :: t()
  def new, do: %__MODULE__{}

  @doc """
  Materializes a profile from a `ChunkAnalysis` (or compatible map) and a
  pre-computed feature vector.

  Fills all 17 primary axes, derives the 4 interaction axes, records
  provenance metadata, and synthesizes the `derived_label`.
  """
  @spec materialize(ChunkAnalysis.t() | map(), list(float())) :: t()
  def materialize(analysis, feature_vector) when is_list(feature_vector) do
    text = Map.get(analysis, :text, "")

    provenance = %{}

    {primary, provenance} = project_hard_copies(analysis, provenance)
    {micro, provenance} = project_micro(feature_vector, text, provenance)
    {composed, provenance} = project_composed(analysis, provenance)

    profile =
      %__MODULE__{}
      |> Map.merge(primary)
      |> Map.merge(micro)
      |> Map.merge(composed)
      |> Map.put(:feature_vector, feature_vector)
      |> Map.put(:feature_provenance, provenance)

    profile
    |> derive_interaction_axes()
    |> derive_label()
  end

  @doc """
  Returns the deterministic label `"domain.speech_act_subtype"`.
  """
  @spec derived_label(t()) :: String.t()
  def derived_label(%__MODULE__{domain: domain, speech_act_subtype: subtype}) do
    "#{domain}.#{subtype}"
  end

  # -------------------------------------------------------------------
  # Hard copies — straight from existing analysis signals
  # -------------------------------------------------------------------

  defp project_hard_copies(analysis, provenance) do
    speech_act = safe_map(analysis, :speech_act)
    discourse = safe_map(analysis, :discourse)
    slots = safe_map(analysis, :slots)

    axes = %{
      speech_act_category: Map.get(speech_act, :category, :unknown),
      speech_act_subtype: Map.get(speech_act, :sub_type, :unknown),
      addressee: Map.get(discourse, :addressee, :unknown),
      slot_completeness: compute_slot_completeness(slots),
      novelty_score: safe_float(analysis, :novelty_score, 0.0),
      confidence: safe_float(analysis, :confidence, 0.0)
    }

    sources =
      [:speech_act_category, :speech_act_subtype, :addressee,
       :slot_completeness, :novelty_score, :confidence]
      |> Enum.into(%{}, &{&1, :analysis})

    {axes, Map.merge(provenance, sources)}
  end

  # -------------------------------------------------------------------
  # MicroClassifier projections
  # -------------------------------------------------------------------

  @micro_axes [
    {:domain, :intent_domain, :unknown},
    {:tense, :tense_class, :present},
    {:aspect, :aspect_class, :simple},
    {:urgency, :urgency, :low},
    {:certainty, :certainty_level, :committed}
  ]

  # Axis classifiers consume the dense feature vector (never text) so that
  # token identity — including proper nouns — cannot influence axis values.
  # See Brain.ML.FeatureVectorClassifier and the noun-invariance regression
  # test in test/brain/analysis/chunk_profile_noun_invariance_test.exs.
  defp project_micro(feature_vector, _text, provenance) do
    {axes, sources} =
      Enum.reduce(@micro_axes, {%{}, %{}}, fn {axis, classifier, default}, {ax, src} ->
        value = safe_classify_vector(classifier, feature_vector, default)
        {Map.put(ax, axis, value), Map.put(src, axis, :micro_classifier)}
      end)

    {axes, Map.merge(provenance, sources)}
  end

  # -------------------------------------------------------------------
  # Composed projections — voting / combining multiple signals
  # -------------------------------------------------------------------

  defp project_composed(analysis, provenance) do
    speech_act = safe_map(analysis, :speech_act)
    discourse = safe_map(analysis, :discourse)
    pos_tags = Map.get(analysis, :pos_tags) || []

    target = derive_target(discourse, pos_tags, analysis)
    modality = derive_modality(speech_act)
    polarity = derive_polarity(pos_tags)
    sentiment_alignment = derive_sentiment_alignment(analysis, speech_act)

    axes = %{
      target: target,
      modality: modality,
      polarity: polarity,
      sentiment_alignment: sentiment_alignment
    }

    sources =
      [:target, :modality, :polarity, :sentiment_alignment]
      |> Enum.into(%{}, &{&1, :composed})

    {axes, Map.merge(provenance, sources)}
  end

  # -------------------------------------------------------------------
  # Interaction axis derivations
  # -------------------------------------------------------------------

  defp derive_interaction_axes(%__MODULE__{} = p) do
    %{p |
      response_posture: derive_response_posture(p),
      engagement_level: derive_engagement_level(p),
      self_disclosure_level: derive_self_disclosure_level(p),
      temporal_framing: derive_temporal_framing(p)
    }
  end

  defp derive_response_posture(%__MODULE__{
         certainty: certainty,
         confidence: confidence,
         slot_completeness: completeness,
         novelty_score: novelty
       }) do
    cond do
      completeness < 0.5 -> :clarify
      certainty in [:speculative, :hedged] -> :hedged
      novelty > 0.7 and confidence < 0.5 -> :tentative_confirm
      confidence < 0.3 -> :hedged
      true -> :direct
    end
  end

  defp derive_engagement_level(%__MODULE__{
         addressee: addressee,
         target: target,
         modality: modality,
         urgency: urgency
       }) do
    cond do
      urgency in [:critical, :high] -> :urgent_demand
      modality == :imperative and target == :agent -> :active_request
      addressee == :bot and modality == :interrogative -> :active_request
      addressee == :unknown and modality == :declarative -> :passive_observation
      true -> :casual_engagement
    end
  end

  defp derive_self_disclosure_level(%__MODULE__{
         target: target,
         domain: domain,
         certainty: certainty,
         sentiment_alignment: sentiment_alignment
       }) do
    cond do
      target != :self -> :none
      sentiment_alignment == :incongruent -> :emotional_self_disclosure
      domain in [:preference, :opinion] -> :preference
      certainty == :tentative -> :opinion
      true -> :factual_self_info
    end
  end

  defp derive_temporal_framing(%__MODULE__{
         tense: tense,
         aspect: aspect,
         polarity: polarity
       }) do
    cond do
      tense == :past and polarity == :negative -> :negated_past
      tense == :past and aspect in [:perfect, :simple] -> :completed_past
      tense == :present and aspect in [:progressive, :perfect_progressive] -> :ongoing
      tense == :future -> :hypothetical_future
      true -> :timeless
    end
  end

  # -------------------------------------------------------------------
  # Label synthesis
  # -------------------------------------------------------------------

  defp derive_label(%__MODULE__{} = p) do
    %{p | derived_label: derived_label(p)}
  end

  # -------------------------------------------------------------------
  # Target derivation — votes from discourse, micro-classifiers, POS
  # -------------------------------------------------------------------

  defp derive_target(discourse, pos_tags, analysis) do
    text = Map.get(analysis, :text, "")
    addressee = Map.get(discourse, :addressee, :unknown)
    directed = safe_classify(:directed_at_bot, text, :unknown)
    personal = safe_classify(:personal_question, text, :unknown)

    pronoun_signal = pronoun_target_signal(pos_tags)

    votes = [
      addressee_vote(addressee),
      directed_vote(directed),
      personal_vote(personal),
      pronoun_signal
    ]

    tally =
      votes
      |> Enum.reject(&is_nil/1)
      |> Enum.reduce(%{}, fn target, acc ->
        Map.update(acc, target, 1, &(&1 + 1))
      end)

    case Enum.max_by(tally, fn {_k, v} -> v end, fn -> {:ambiguous, 0} end) do
      {winner, count} when count >= 2 -> winner
      _ -> :ambiguous
    end
  end

  defp addressee_vote(:bot), do: :agent
  defp addressee_vote(:user), do: :self
  defp addressee_vote(:third_party), do: :other_person
  defp addressee_vote(_), do: nil

  defp directed_vote(label) when label in [:yes, "yes", :directed, "directed"], do: :agent
  defp directed_vote(_), do: nil

  defp personal_vote(label) when label in [:personal, "personal", :yes, "yes"], do: :agent
  defp personal_vote(_), do: nil

  defp pronoun_target_signal(pos_tags) do
    tokens =
      pos_tags
      |> Enum.map(fn
        {word, _tag} -> String.downcase(word)
        %{word: word} -> String.downcase(word)
        _ -> nil
      end)
      |> Enum.reject(&is_nil/1)

    first_person = Enum.count(tokens, &(&1 in ~w(i me my mine myself)))
    second_person = Enum.count(tokens, &(&1 in ~w(you your yours yourself yourselves)))

    cond do
      first_person > second_person and first_person > 0 -> :self
      second_person > first_person and second_person > 0 -> :agent
      true -> nil
    end
  end

  # -------------------------------------------------------------------
  # Modality derivation — from speech-act booleans
  # -------------------------------------------------------------------

  defp derive_modality(speech_act) do
    is_question = Map.get(speech_act, :is_question, false)
    is_imperative = Map.get(speech_act, :is_imperative, false)
    category = Map.get(speech_act, :category, :unknown)

    cond do
      is_question == true -> :interrogative
      is_imperative == true -> :imperative
      category == :expressive -> :exclamatory
      true -> :declarative
    end
  end

  # -------------------------------------------------------------------
  # Polarity derivation — negation particle count from POS tags
  # -------------------------------------------------------------------

  defp derive_polarity(pos_tags) do
    neg_count =
      Enum.count(pos_tags, fn
        {_word, :PART} -> true
        %{tag: :PART} -> true
        _ -> false
      end)

    if neg_count > 0, do: :negative, else: :affirmative
  end

  # -------------------------------------------------------------------
  # Sentiment alignment — congruence between sentiment and speech act
  # -------------------------------------------------------------------

  defp derive_sentiment_alignment(analysis, speech_act) do
    sentiment = Map.get(analysis, :sentiment)
    label = sentiment_label(sentiment)
    category = Map.get(speech_act, :category, :unknown)

    expected = expected_sentiment(category)

    cond do
      label == :unknown or expected == :any -> :neutral
      label == expected -> :congruent
      true -> :incongruent
    end
  end

  defp sentiment_label(%{label: label}) when is_atom(label), do: label
  defp sentiment_label(%{label: label}) when is_binary(label), do: safe_to_atom(label)
  defp sentiment_label(_), do: :unknown

  defp expected_sentiment(:expressive), do: :positive
  defp expected_sentiment(:directive), do: :any
  defp expected_sentiment(:commissive), do: :positive
  defp expected_sentiment(_), do: :any

  # -------------------------------------------------------------------
  # Slot completeness
  # -------------------------------------------------------------------

  defp compute_slot_completeness(%{} = slots) do
    filled = slots |> Map.get(:filled_slots, %{}) |> map_size()
    missing = slots |> Map.get(:missing_required, []) |> length()
    total = filled + missing

    if total == 0, do: 1.0, else: filled / total
  end

  # -------------------------------------------------------------------
  # Safe MicroClassifier wrapper
  # -------------------------------------------------------------------

  defp safe_classify(classifier, text, default) do
    case MicroClassifiers.classify(classifier, text) do
      {:ok, label, _score} -> safe_to_atom(label)
      _ -> default
    end
  rescue
    _ -> default
  catch
    :exit, _ -> default
  end

  defp safe_classify_vector(classifier, feature_vector, default) do
    case MicroClassifiers.classify_vector(classifier, feature_vector) do
      {:ok, label, _score} -> safe_to_atom(label)
      _ -> default
    end
  rescue
    _ -> default
  catch
    :exit, _ -> default
  end

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------

  defp safe_map(analysis, key) do
    case Map.get(analysis, key) do
      m when is_map(m) -> m
      _ -> %{}
    end
  end

  defp safe_float(analysis, key, default) do
    case Map.get(analysis, key) do
      v when is_number(v) -> v * 1.0
      _ -> default
    end
  end

  defp safe_to_atom(value) when is_binary(value) do
    String.to_existing_atom(value)
  rescue
    ArgumentError -> String.to_atom(value)
  end
end
