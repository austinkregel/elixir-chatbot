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

  ## Provenance: computed vs defaulted

  Every axis records how its value was arrived at, in `:feature_provenance`.
  This matters because an axis at its default value and an axis genuinely
  computed to that same value were previously indistinguishable: a classifier
  that failed to load returned `:low` for urgency, and so did a classifier
  that read the input and decided `:low`.

  That ambiguity blocks two things task 082 needs. A snapshot cannot be
  compared across runs if a value might be a placeholder, and the corpus
  selection rule in task 083 — *keep a sentence when at least one axis is off
  its default and genuinely computed* — is not checkable at all.

  Each entry is a map:

      %{source: :micro_classifier, status: :computed,  classifier: :urgency, confidence: 0.83}
      %{source: :micro_classifier, status: :defaulted, classifier: :urgency, reason: :not_loaded}
      %{source: :analysis,         status: :defaulted, field: :novelty_score, reason: :absent}
      %{source: :composed,         status: :defaulted, reason: :no_majority, evidence: %{...}}
      %{source: :derived,          status: :computed,  depends_on: [:tense, :aspect, :polarity]}

  `:status` is the load-bearing field. `:defaulted` means the value in the
  struct is the declared default because the axis could not be determined —
  not that it happens to equal the default. `:reason` says why, and for
  composed axes `:evidence` carries the inputs the derivation saw, so a wrong
  value can be traced to the signal that produced it rather than guessed at.

  ## On the remaining fallbacks

  `safe_classify_vector/3` still substitutes a default rather than raising when
  a classifier is unavailable. That is deliberate for now and narrowly scoped:
  a `:wrong_kind` result raises, because asking a text classifier for a vector
  classification is a wiring bug rather than a data condition. `:not_loaded`
  and `:classification_failed` are recorded as `:defaulted` with the reason
  attached, which makes them reportable instead of invisible. Measured
  2026-09-12: 30 of 30 classifier calls answered, so none of these paths is
  currently taken.
  """

  alias Brain.Analysis.ChunkAnalysis
  alias Brain.LinguisticData
  alias Brain.ML.Tokenizer
  alias Brain.ML.MicroClassifiers

  @type t :: %__MODULE__{
          domain: atom(),
          speech_act_category: atom(),
          speech_act_subtype: atom(),
          target: atom(),
          modality: atom(),
          polarity: float(),
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
            polarity: 0.0,
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

  # Declared shape of all 18 axes: {axis, kind, domain, source}.
  #
  #   kind   — :categorical or :continuous
  #   domain — {:enum, [atom]} for a closed set declared here,
  #            {:model, classifier} for a set owned by a trained model,
  #            {:range, lo, hi} for a continuous axis
  #   source — how the axis is filled: :analysis, :micro_classifier,
  #            :composed, or :derived (from other axes)
  #
  # `{:model, _}` is not a cop-out. For a classifier-backed axis the trained
  # model owns the value set, and restating it here would create a copy that
  # goes stale on the next retrain. `axis_domain/1` resolves it by asking the
  # model, so the declaration cannot drift from what the classifier can emit.
  #
  # `:enum` sets are transcribed from this module's own `cond` branches, which
  # are the only thing that can produce those values.
  @axis_manifest [
    {:domain, :categorical, {:model, :intent_domain}, :micro_classifier},
    {:tense, :categorical, {:model, :tense_class}, :micro_classifier},
    {:aspect, :categorical, {:model, :aspect_class}, :micro_classifier},
    {:urgency, :categorical, {:model, :urgency}, :micro_classifier},
    {:certainty, :categorical, {:model, :certainty_level}, :micro_classifier},
    {:speech_act_category, :categorical,
     {:enum, [:assertive, :directive, :commissive, :expressive, :declarative, :unknown]},
     :analysis},
    {:speech_act_subtype, :categorical, {:open, :speech_act_subtype}, :analysis},
    {:addressee, :categorical, {:enum, [:bot, :user, :third_party, :ambiguous, :unknown]},
     :analysis},
    {:slot_completeness, :continuous, {:range, 0.0, 1.0}, :analysis},
    {:novelty_score, :continuous, {:range, 0.0, 1.0}, :analysis},
    {:target, :categorical, {:enum, [:agent, :self, :other_person, :ambiguous]}, :composed},
    {:modality, :categorical, {:enum, [:interrogative, :imperative, :exclamatory, :declarative]},
     :composed},
    {:polarity, :continuous, {:range, 0.0, 1.0}, :composed},
    {:sentiment_alignment, :categorical, {:enum, [:neutral, :congruent, :incongruent]},
     :composed},
    {:response_posture, :categorical, {:enum, [:clarify, :hedged, :tentative_confirm, :direct]},
     :derived},
    {:engagement_level, :categorical,
     {:enum, [:urgent_demand, :active_request, :passive_observation, :casual_engagement]},
     :derived},
    {:self_disclosure_level, :categorical,
     {:enum, [:none, :emotional_self_disclosure, :preference, :opinion, :factual_self_info]},
     :derived},
    {:temporal_framing, :categorical,
     {:enum, [:negated_past, :completed_past, :ongoing, :hypothetical_future, :timeless]},
     :derived}
  ]

  @doc """
  Returns the declared shape of all 18 axes as `{axis, kind, domain, source}`.

  The counterpart to `ChunkFeatures.dimension_manifest/0`, one layer up: that
  names the 343 feature dimensions, this declares what the 18 axes projected
  from them are allowed to contain.

  Needed because "is this axis correct?" is unanswerable without first knowing
  what values it may legitimately take. Task 081 measured four axes stuck on a
  single value; distinguishing *stuck* from *correctly constant* requires a
  declared domain to compare against.

  `{:model, classifier}` domains are owned by a trained model rather than
  restated here — see `axis_domain/1`.
  """
  @spec axis_manifest() :: [{atom(), :categorical | :continuous, tuple(), atom()}]
  def axis_manifest, do: @axis_manifest

  @doc """
  Returns the list of axis names, in manifest order.
  """
  @spec axes() :: [atom()]
  def axes, do: Enum.map(@axis_manifest, fn {axis, _k, _d, _s} -> axis end)

  @doc """
  Resolves one axis's value domain to a concrete set or range.

  For `{:model, classifier}` axes this asks `MicroClassifiers.labels/1`, so the
  answer is what the loaded model can actually emit. Returns:

    * `{:enum, [atom]}` — a closed set of permitted values
    * `{:range, lo, hi}` — a continuous interval
    * `{:open, reason}` — deliberately unconstrained (see below)
    * `{:error, reason}` — a model-backed domain that could not be read

  `{:open, :speech_act_subtype}` is honest rather than lazy: subtypes are
  copied straight from whatever the speech-act analyzer produced, and no
  canonical list exists yet. Closing it is task 077.

  A `{:model, _}` axis reports `{:error, reason}` instead of falling back to a
  guessed set — an axis whose permitted values are unknown must not be
  validated against an invented domain, because every value would then appear
  to pass.
  """
  @spec axis_domain(atom()) ::
          {:enum, [atom()]} | {:range, float(), float()} | {:open, atom()} | {:error, atom()}
  def axis_domain(axis) when is_atom(axis) do
    case Enum.find(@axis_manifest, fn {a, _k, _d, _s} -> a == axis end) do
      nil ->
        {:error, :unknown_axis}

      {_a, _k, {:model, classifier}, _s} ->
        case MicroClassifiers.labels(classifier) do
          {:ok, labels} -> {:enum, Enum.map(labels, &safe_to_atom/1)}
          {:error, reason} -> {:error, reason}
        end

      {_a, _k, domain, _s} ->
        domain
    end
  end

  @doc """
  Returns the axis's declared default — the value it holds when undetermined.

  Read from `%ChunkProfile{}` itself so it cannot disagree with the struct.
  Paired with `:feature_provenance`, this is what makes task 083's selection
  rule checkable: an axis counts as a demonstrator when it is off this value
  *and* its provenance says `:computed`.
  """
  @spec axis_default(atom()) :: term()
  def axis_default(axis) when is_atom(axis) do
    Map.fetch!(%__MODULE__{}, axis)
  end

  @doc """
  Returns the provenance entry for one axis, or `nil` when the profile has none.

  Every entry carries at least `:source` and `:status`. Which further keys are
  present depends on the source: `:micro_classifier` entries add `:classifier`
  and `:confidence`, `:analysis` entries add `:field`, `:composed` entries add
  `:evidence`, `:derived` entries add `:depends_on` and `:parents_defaulted`,
  and every `:defaulted` entry adds `:reason`.

  Callers previously reached into `:feature_provenance` directly, three
  different ways across three call sites — `Map.get/2`, `get_in/2` and a
  `%{status: :defaulted}` pattern match. This is the one accessor, added before
  a fourth appeared.
  """
  @spec provenance(t(), atom()) :: map() | nil
  def provenance(%__MODULE__{feature_provenance: provenance}, axis) when is_atom(axis) do
    Map.get(provenance, axis)
  end

  @doc """
  Whether an axis was actually determined, as opposed to left at its default.

  An unrecorded axis is `false`: absent provenance is not evidence that the
  axis was computed. Note this is weaker than task 083's selection rule, which
  also requires the value to be off `axis_default/1` — an axis can be genuinely
  computed and land on its default value, and the two claims are different.
  """
  @spec computed?(t(), atom()) :: boolean()
  def computed?(%__MODULE__{} = profile, axis) when is_atom(axis) do
    case provenance(profile, axis) do
      %{status: :computed} -> true
      _ -> false
    end
  end

  @doc """
  Why an axis holds its default, or `nil` when it was computed or unrecorded.

  The distinction this preserves: a `:defaulted` axis holds exactly its
  declared default, so the value alone cannot say whether the axis was
  determined to be that value or never determined at all.
  """
  @spec default_reason(t(), atom()) :: atom() | nil
  def default_reason(%__MODULE__{} = profile, axis) when is_atom(axis) do
    case provenance(profile, axis) do
      %{status: :defaulted, reason: reason} -> reason
      _ -> nil
    end
  end

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

    # Each axis reports whether the analysis actually supplied its field. An
    # absent field yields the declared default, and saying so is the whole
    # point: `novelty_score` is read here but written nowhere in the analysis
    # pipeline (task 073), so it is `:defaulted` on every chunk — a fact that
    # was invisible while provenance recorded only `:analysis`.
    {category, category_prov} = copied(speech_act, :category, :unknown)
    {subtype, subtype_prov} = copied(speech_act, :sub_type, :unknown)
    {addressee, addressee_prov} = copied(discourse, :addressee, :unknown)
    {completeness, completeness_prov} = slot_completeness_with_provenance(slots)
    {novelty, novelty_prov} = copied_float(analysis, :novelty_score, 0.0)
    {confidence, confidence_prov} = copied_float(analysis, :confidence, 0.0)

    axes = %{
      speech_act_category: category,
      speech_act_subtype: subtype,
      addressee: addressee,
      slot_completeness: completeness,
      novelty_score: novelty,
      confidence: confidence
    }

    sources = %{
      speech_act_category: category_prov,
      speech_act_subtype: subtype_prov,
      addressee: addressee_prov,
      slot_completeness: completeness_prov,
      novelty_score: novelty_prov,
      confidence: confidence_prov
    }

    {axes, Map.merge(provenance, sources)}
  end

  # Copies an atom-valued field out of an analysis sub-map, recording whether
  # it was present. `nil` counts as absent: a field explicitly set to nil
  # carries no more information than a missing one.
  defp copied(source_map, field, default) do
    case Map.get(source_map, field) do
      nil ->
        {default, %{source: :analysis, status: :defaulted, field: field, reason: :absent}}

      value ->
        {value, %{source: :analysis, status: :computed, field: field}}
    end
  end

  defp copied_float(analysis, field, default) do
    case Map.get(analysis, field) do
      v when is_number(v) ->
        {v * 1.0, %{source: :analysis, status: :computed, field: field}}

      nil ->
        {default, %{source: :analysis, status: :defaulted, field: field, reason: :absent}}

      other ->
        {default,
         %{
           source: :analysis,
           status: :defaulted,
           field: field,
           reason: :not_a_number,
           evidence: %{got: inspect(other)}
         }}
    end
  end

  # `compute_slot_completeness/1` returns 1.0 both for "every required slot is
  # filled" and for "no slots were detected at all". Task 081 measured this
  # axis at 1.0 for 99% of a corpus containing fragments like "also the air
  # conditioner", which is not a plausible reading — the second case was
  # masquerading as the first. They are now distinguishable.
  defp slot_completeness_with_provenance(slots) do
    filled = slots |> Map.get(:filled_slots, %{}) |> map_size()
    missing = slots |> Map.get(:missing_required, []) |> length()
    total = filled + missing

    if total == 0 do
      {1.0,
       %{
         source: :analysis,
         status: :defaulted,
         field: :slots,
         reason: :no_slots_detected,
         evidence: %{filled: filled, missing_required: missing}
       }}
    else
      {filled / total,
       %{
         source: :analysis,
         status: :computed,
         field: :slots,
         evidence: %{filled: filled, missing_required: missing}
       }}
    end
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
        {value, prov} = classify_vector_with_provenance(classifier, feature_vector, default)
        {Map.put(ax, axis, value), Map.put(src, axis, prov)}
      end)

    {axes, Map.merge(provenance, sources)}
  end

  # Records whether the classifier answered, and on failure why. The confidence
  # is kept on a computed value because an axis answered at 0.31 and one
  # answered at 0.95 are not equally trustworthy, and a snapshot that omits it
  # cannot tell them apart later.
  defp classify_vector_with_provenance(classifier, feature_vector, default) do
    case classify_vector_result(classifier, feature_vector) do
      {:ok, label, score} ->
        {safe_to_atom(label),
         %{
           source: :micro_classifier,
           status: :computed,
           classifier: classifier,
           confidence: score
         }}

      {:error, reason} ->
        {default,
         %{
           source: :micro_classifier,
           status: :defaulted,
           classifier: classifier,
           reason: reason
         }}
    end
  end

  # `:wrong_kind` is not a data condition — it means a text-backed classifier
  # was wired into a vector-classified axis, which no input can fix and which
  # would otherwise show up as an axis that is merely always defaulted. It
  # raises rather than being recorded.
  defp classify_vector_result(classifier, feature_vector) do
    case MicroClassifiers.classify_vector(classifier, feature_vector) do
      {:ok, label, score} ->
        {:ok, label, score}

      {:error, :wrong_kind} ->
        raise ArgumentError, """
        #{inspect(classifier)} is a text classifier but is registered as a
        feature-vector axis in ChunkProfile's @micro_axes. This is a wiring
        error, not a missing model — no input can make it succeed.
        """

      {:error, reason} ->
        {:error, reason}
    end
  rescue
    e in ArgumentError -> reraise e, __STACKTRACE__
    e -> {:error, {:raised, e.__struct__}}
  catch
    :exit, _ -> {:error, :classifier_exited}
  end

  # -------------------------------------------------------------------
  # Composed projections — voting / combining multiple signals
  # -------------------------------------------------------------------

  defp project_composed(analysis, provenance) do
    speech_act = safe_map(analysis, :speech_act)
    discourse = safe_map(analysis, :discourse)
    pos_tags = Map.get(analysis, :pos_tags) || []

    {target, target_prov} = derive_target(discourse, pos_tags, analysis)
    {modality, modality_prov} = derive_modality(speech_act)
    {polarity, polarity_prov} = derive_polarity(pos_tags, Map.get(analysis, :text, ""))
    {sentiment_alignment, sentiment_prov} = derive_sentiment_alignment(analysis, speech_act)

    axes = %{
      target: target,
      modality: modality,
      polarity: polarity,
      sentiment_alignment: sentiment_alignment
    }

    sources = %{
      target: target_prov,
      modality: modality_prov,
      polarity: polarity_prov,
      sentiment_alignment: sentiment_prov
    }

    {axes, Map.merge(provenance, sources)}
  end

  # -------------------------------------------------------------------
  # Interaction axis derivations
  # -------------------------------------------------------------------

  defp derive_interaction_axes(%__MODULE__{} = p) do
    {posture, posture_prov} = derive_response_posture(p)
    {engagement, engagement_prov} = derive_engagement_level(p)
    {disclosure, disclosure_prov} = derive_self_disclosure_level(p)
    {framing, framing_prov} = derive_temporal_framing(p)

    provenance =
      Map.merge(p.feature_provenance, %{
        response_posture: posture_prov,
        engagement_level: engagement_prov,
        self_disclosure_level: disclosure_prov,
        temporal_framing: framing_prov
      })

    %{
      p
      | response_posture: posture,
        engagement_level: engagement,
        self_disclosure_level: disclosure,
        temporal_framing: framing,
        feature_provenance: provenance
    }
  end

  # Interaction axes read other axes rather than the analysis, so their
  # provenance names the parents. `:parents_defaulted` lists which of those
  # parents were themselves defaulted: an interaction axis computed from
  # placeholder inputs is not meaningfully computed, and task 081's warning
  # that derived axes "re-count their parents' information" applies to their
  # failures too.
  defp derived_prov(status, parents, profile, extra \\ %{}) do
    # Matches `:defaulted` rather than `not computed?/2`: those differ for a
    # parent with no provenance entry at all, and this counts only parents
    # positively recorded as defaulted. Every parent is recorded today, so the
    # two agree in practice; keeping the narrower test avoids changing what
    # `parents_defaulted` means as a side effect of introducing the accessor.
    defaulted =
      Enum.filter(parents, fn parent ->
        match?(%{status: :defaulted}, provenance(profile, parent))
      end)

    Map.merge(
      %{
        source: :derived,
        status: status,
        depends_on: parents,
        parents_defaulted: defaulted
      },
      extra
    )
  end

  @response_posture_parents [:certainty, :confidence, :slot_completeness, :novelty_score]

  defp derive_response_posture(
         %__MODULE__{
           certainty: certainty,
           confidence: confidence,
           slot_completeness: completeness,
           novelty_score: novelty
         } = p
       ) do
    parents = @response_posture_parents

    cond do
      completeness < 0.5 ->
        {:clarify, derived_prov(:computed, parents, p)}

      # NOTE: `:speculative` is not in certainty_level's model vocabulary
      # (["committed", "hedged", "tentative"]), so only the `:hedged` half of
      # this test can ever fire. Recorded, not fixed — see the commit adding
      # MicroClassifiers.labels/1.
      certainty in [:speculative, :hedged] ->
        {:hedged, derived_prov(:computed, parents, p)}

      novelty > 0.7 and confidence < 0.5 ->
        {:tentative_confirm, derived_prov(:computed, parents, p)}

      confidence < 0.3 ->
        {:hedged, derived_prov(:computed, parents, p)}

      true ->
        {:direct, derived_prov(:defaulted, parents, p, %{reason: :no_posture_signal})}
    end
  end

  @engagement_parents [:addressee, :target, :modality, :urgency]

  defp derive_engagement_level(
         %__MODULE__{
           addressee: addressee,
           target: target,
           modality: modality,
           urgency: urgency
         } = p
       ) do
    parents = @engagement_parents

    cond do
      urgency in [:critical, :high] ->
        {:urgent_demand, derived_prov(:computed, parents, p)}

      modality == :imperative and target == :agent ->
        {:active_request, derived_prov(:computed, parents, p)}

      addressee == :bot and modality == :interrogative ->
        {:active_request, derived_prov(:computed, parents, p)}

      addressee == :unknown and modality == :declarative ->
        {:passive_observation, derived_prov(:computed, parents, p)}

      true ->
        {:casual_engagement,
         derived_prov(:defaulted, parents, p, %{reason: :no_engagement_signal})}
    end
  end

  @disclosure_parents [:target, :domain, :certainty, :sentiment_alignment]

  defp derive_self_disclosure_level(
         %__MODULE__{
           target: target,
           domain: domain,
           certainty: certainty,
           sentiment_alignment: sentiment_alignment
         } = p
       ) do
    parents = @disclosure_parents

    cond do
      # Task 081: `target` never reaches `:self`, so this short-circuit is the
      # only outcome in practice. It is a defaulted `:none`, not a judgement
      # that nothing was disclosed.
      target != :self ->
        {:none, derived_prov(:defaulted, parents, p, %{reason: :target_not_self})}

      sentiment_alignment == :incongruent ->
        {:emotional_self_disclosure, derived_prov(:computed, parents, p)}

      # NOTE: intent_domain's vocabulary contains neither "preference" nor
      # "opinion", so this branch is unreachable regardless of `target`.
      domain in [:preference, :opinion] ->
        {:preference, derived_prov(:computed, parents, p)}

      certainty == :tentative ->
        {:opinion, derived_prov(:computed, parents, p)}

      true ->
        {:factual_self_info, derived_prov(:computed, parents, p)}
    end
  end

  # Polarity is a negation STRENGTH on 0.0..1.0, not a flag. These two points
  # on that scale are named because both derive_polarity/2 (which assigns them)
  # and derive_temporal_framing/1 (which thresholds on them) must agree.
  #
  # sentential: negation scoping over the clause -- "I am not competent".
  # constituent: negation inside a phrase while the clause still asserts --
  #   "the unhealthy meals I cook". Slightly negative, not negative.
  @sentential_negation 1.0
  @constituent_negation 0.35

  @framing_parents [:tense, :aspect, :polarity]

  defp derive_temporal_framing(
         %__MODULE__{
           tense: tense,
           aspect: aspect,
           polarity: polarity
         } = p
       ) do
    parents = @framing_parents

    cond do
      tense == :past and polarity >= @sentential_negation ->
        {:negated_past, derived_prov(:computed, parents, p)}

      tense == :past and aspect in [:perfect, :simple] ->
        {:completed_past, derived_prov(:computed, parents, p)}

      tense == :present and aspect in [:progressive, :perfect_progressive] ->
        {:ongoing, derived_prov(:computed, parents, p)}

      tense == :future ->
        {:hypothetical_future, derived_prov(:computed, parents, p)}

      true ->
        {:timeless, derived_prov(:defaulted, parents, p, %{reason: :no_temporal_signal})}
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

    # The vote breakdown is carried into provenance because `:ambiguous` is
    # both this axis's default and a legitimate verdict. Task 081 found
    # `target` never reaches `:self` — recording which voters fired makes the
    # cause visible per chunk instead of requiring a separate investigation:
    # `:self` needs two concurring votes, and only `addressee_vote(:user)` and
    # the pronoun signal can supply one.
    evidence = %{
      addressee: addressee,
      directed: directed,
      personal: personal,
      pronoun_signal: pronoun_signal,
      tally: tally
    }

    case Enum.max_by(tally, fn {_k, v} -> v end, fn -> {:ambiguous, 0} end) do
      {winner, count} when count >= 2 ->
        {winner,
         %{
           source: :composed,
           status: :computed,
           votes: count,
           evidence: evidence
         }}

      {_winner, count} ->
        {:ambiguous,
         %{
           source: :composed,
           status: :defaulted,
           reason: :no_majority,
           votes: count,
           evidence: evidence
         }}
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
    tokens = pos_tokens(pos_tags)

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

    evidence = %{is_question: is_question, is_imperative: is_imperative, category: category}

    cond do
      is_question == true ->
        {:interrogative, %{source: :composed, status: :computed, evidence: evidence}}

      is_imperative == true ->
        {:imperative, %{source: :composed, status: :computed, evidence: evidence}}

      category == :expressive ->
        {:exclamatory, %{source: :composed, status: :computed, evidence: evidence}}

      true ->
        # `:declarative` here is the absence of any positive signal, not a
        # detected declarative — the two were previously indistinguishable.
        {:declarative,
         %{
           source: :composed,
           status: :defaulted,
           reason: :no_modality_signal,
           evidence: evidence
         }}
    end
  end

  # -------------------------------------------------------------------
  # Polarity derivation — negation particle count from POS tags
  # -------------------------------------------------------------------

  # Polarity is the STRENGTH of negation, not a yes/no.
  #
  # It was an enum of [:affirmative, :negative] decided by counting PART tags.
  # Two things were wrong with that.
  #
  # First, English does not put negation only on PART. Measured over 206 negated
  # sentences, 108 carry it on PART ("not", "n't") and the rest on ADV
  # ("never"), DET ("no") or PRON ("nothing") — so a PART count is blind to
  # 47.1% of negated input even with a correct tagger, and the current tagger
  # emits no PART at all.
  #
  # Second, negation is not binary. "I am not competent" negates the predicate.
  # "the unhealthy meals I cook" carries negation inside a noun phrase while the
  # clause asserts something positively; it is slightly negative, not negative.
  # Collapsing both to :negative loses the distinction that matters downstream.
  #
  # Scale: 0.0 affirmative, 1.0 fully negated.
  defp derive_polarity(pos_tags, text) do
    tokens =
      text
      |> Tokenizer.expand_contractions()
      |> Tokenizer.tokenize_normalized(expand_contractions: false)

    # Closed-class negators scope over the clause. They are function words, so
    # WordNet cannot supply them; they come from the declared vocabulary.
    sentential = Enum.filter(tokens, &LinguisticData.negation?/1)

    # Morphological negators ("unable", "useless") are derived from WordNet
    # antonymy plus a negative affix. They negate the word they attach to, not
    # necessarily the clause, so they score lower and accumulate.
    constituent = Enum.filter(tokens, &LinguisticData.morphological_negator?/1)

    score =
      cond do
        sentential != [] -> @sentential_negation
        constituent != [] -> min(@constituent_negation * length(constituent), 1.0)
        true -> 0.0
      end

    atom_part =
      Enum.count(pos_tags, fn
        {_word, :PART} -> true
        %{tag: :PART} -> true
        _ -> false
      end)

    string_part =
      Enum.count(pos_tags, fn
        {_word, "PART"} -> true
        %{tag: "PART"} -> true
        _ -> false
      end)

    evidence = %{
      pos_tag_count: length(pos_tags),
      atom_part: atom_part,
      string_part: string_part,
      sentential_negators: sentential,
      constituent_negators: constituent
    }

    if tokens == [] do
      {0.0, %{source: :composed, status: :defaulted, reason: :no_tokens, evidence: evidence}}
    else
      {score, %{source: :composed, status: :computed, evidence: evidence}}
    end
  end

  # Lowercased surface tokens from a POS-tagged list. Tolerates the tuple and
  # map shapes the tagger and its callers both produce.
  defp pos_tokens(pos_tags) do
    pos_tags
    |> Enum.map(fn
      {word, _tag} when is_binary(word) -> String.downcase(word)
      %{word: word} when is_binary(word) -> String.downcase(word)
      %{token: word} when is_binary(word) -> String.downcase(word)
      _ -> nil
    end)
    |> Enum.reject(&is_nil/1)
  end

  # -------------------------------------------------------------------
  # Sentiment alignment — congruence between sentiment and speech act
  # -------------------------------------------------------------------

  defp derive_sentiment_alignment(analysis, speech_act) do
    sentiment = Map.get(analysis, :sentiment)
    label = sentiment_label(sentiment)
    category = Map.get(speech_act, :category, :unknown)

    expected = expected_sentiment(category)

    evidence = %{sentiment_label: label, speech_act_category: category, expected: expected}

    cond do
      # Task 081: across 238 utterances the speech-act classifier produced only
      # :assertive and :directive, both of which map to `expected == :any`, so
      # this branch always won and the axis was constant :neutral. Recording it
      # as defaulted — with the category that caused it — means the axis stops
      # claiming to have compared anything.
      label == :unknown or expected == :any ->
        {:neutral,
         %{
           source: :composed,
           status: :defaulted,
           reason: if(label == :unknown, do: :no_sentiment, else: :no_expectation_for_category),
           evidence: evidence
         }}

      label == expected ->
        {:congruent, %{source: :composed, status: :computed, evidence: evidence}}

      true ->
        {:incongruent, %{source: :composed, status: :computed, evidence: evidence}}
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
  # Safe MicroClassifier wrapper
  # -------------------------------------------------------------------

  # Still used by `derive_target/3` for the two text-backed votes. Unlike the
  # vector path it does not yet distinguish a defaulted vote from a computed
  # one; the vote breakdown in `target`'s provenance carries the raw labels
  # instead, which is enough to see which voters fired.
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

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------

  defp safe_map(analysis, key) do
    case Map.get(analysis, key) do
      m when is_map(m) -> m
      _ -> %{}
    end
  end

  defp safe_to_atom(value) when is_binary(value) do
    String.to_existing_atom(value)
  rescue
    ArgumentError -> String.to_atom(value)
  end
end
