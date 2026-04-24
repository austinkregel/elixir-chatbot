defmodule Brain.Analysis.FeatureExtractor.ChunkFeatures do
  @moduledoc """
  Aggregates per-word features and existing analysis signals into a
  per-chunk feature vector spanning 22 groups.

  ## Feature Groups

  1. Surface/lexical
  2. POS distribution
  3. Syntactic/structural
  4. Pronoun/person/target
  5. Modality/certainty
  6. Speech act
  7. Discourse
  8. Sentiment
  9. Entity
  10. Lexical-semantic fingerprint (legacy merged supersense distribution)
  11. Word-meaning depth/ambiguity
  12. Semantic-role frames
  13. Memory/context
  14. Slot completeness
  15. **Wh-target type** *(Tier 1 enrichment, identity-invariant)*
  16. **Time-expression typology** *(Tier 1, identity-invariant)*
  17. **POS-conditional supersense fingerprints (verb / noun / adj-adv)** *(Tier 1)*
  18. **ConceptNet edge-type fingerprint** *(Tier 1)*
  19. **Verb × argument selectional-preference cross-product** *(Tier 1)*
  20. **Subcategorization frame** *(Tier 2, POS-only — identity-invariant)*
  21. **Discourse-marker semantic categories** *(Tier 2, closed-class lexical)*
  22. **Speech-act × wh-target interaction grid** *(Tier 2, materializes joint
      that the linear centroid cannot learn on its own)*

  Groups 15-22 are produced by
  `Brain.Analysis.FeatureExtractor.EnrichmentFeatures` and are
  *appended at the tail* of the vector so that pre-existing offsets
  for groups 1-14 are not perturbed. See that module's `@moduledoc`
  for the linguistic motivation and citations.
  """

  alias Brain.Analysis.FeatureExtractor.{WordFeatures, EnrichmentFeatures}
  alias Brain.Lexicon

  @all_pos [:NOUN, :PROPN, :VERB, :AUX, :ADJ, :ADV, :PRON, :DET, :ADP, :CONJ, :PART, :NUM, :INTJ, :PUNCT, :SYM, :X]

  @speech_act_categories [:assertive, :directive, :commissive, :expressive, :declarative, :unknown]
  @question_subtypes [:request_information, :request_action, :command]
  @addressee_values [:bot, :user, :third_party, :unknown]

  @pronoun_1st_singular ~w(i me my mine myself)
  @pronoun_1st_plural ~w(we us our ours ourselves)
  @pronoun_2nd ~w(you your yours yourself yourselves)
  @pronoun_3rd ~w(he she him her his hers they them their theirs himself herself themselves it its itself)
  @pronoun_demonstrative ~w(this that these those)
  @pronoun_interrogative ~w(who whom whose which what)
  @pronoun_reflexive ~w(myself yourself himself herself itself ourselves yourselves themselves)
  @pronoun_possessive ~w(my your his her its our their mine yours hers ours theirs)

  @entity_types [:person, :location, :organization, :date, :number, :topic, :device, :concept]

  @srl_roles [:agent, :patient, :instrument, :location, :time, :manner, :recipient, :cause]

  @doc """
  Extracts a full chunk feature vector from a ChunkAnalysis struct
  and its per-word features.

  Returns a flat list of floats representing the ~140-dimension vector.
  """
  def extract(%{} = analysis, word_features \\ nil) do
    tokens = extract_tokens(analysis)
    pos_tags = Map.get(analysis, :pos_tags) || []

    word_feats = word_features || WordFeatures.extract(pos_tags)
    content_word_tokens = content_word_tokens(word_feats)

    group1 = surface_lexical(tokens, word_feats)
    group2 = pos_distribution(pos_tags)
    group3 = syntactic_structural(tokens, pos_tags)
    group4 = pronoun_person(tokens)
    group5 = modality_certainty(tokens, pos_tags)
    group6 = speech_act_features(analysis)
    group7 = discourse_features(analysis)
    group8 = sentiment_features(analysis)
    group9 = entity_features(analysis)
    group10 = lexical_semantic_fingerprint(word_feats)
    group11 = word_meaning_depth(word_feats)
    group12 = srl_frame_features(analysis)
    group13 = memory_context_features(analysis)
    group14 = slot_completeness_features(analysis)

    # Enrichment groups (Tier 1). Appended at the tail so existing
    # offsets are preserved.
    group15 = EnrichmentFeatures.wh_target(tokens)
    group16 = EnrichmentFeatures.time_typology(tokens)
    group17_verb = EnrichmentFeatures.verb_supersenses(word_feats)
    group17_noun = EnrichmentFeatures.noun_supersenses(word_feats)
    group17_adj = EnrichmentFeatures.adj_adv_supersenses(word_feats)
    group18 = EnrichmentFeatures.conceptnet_edges(content_word_tokens)
    group19 = EnrichmentFeatures.selectional_preferences(word_feats)

    # Enrichment groups (Tier 2). Same tail-append discipline.
    group20 = EnrichmentFeatures.subcategorization_frame(token_pos_pairs(tokens, pos_tags))
    group21 = EnrichmentFeatures.discourse_markers(tokens)

    group22 =
      EnrichmentFeatures.speech_act_wh_interaction(
        speech_act_label_for_interaction(analysis),
        tokens
      )

    group1 ++ group2 ++ group3 ++ group4 ++ group5 ++ group6 ++
      group7 ++ group8 ++ group9 ++ group10 ++ group11 ++ group12 ++
      group13 ++ group14 ++
      group15 ++ group16 ++ group17_verb ++ group17_noun ++ group17_adj ++
      group18 ++ group19 ++
      group20 ++ group21 ++ group22
  end

  # Builds `[{token, pos}]` pairs for `subcategorization_frame/1`. The
  # subcat group reads only the POS dimension, so missing/short pos_tag
  # alignment degrades gracefully (the trailing tokens get :X, which
  # contributes nothing to any of the frame detectors).
  defp token_pos_pairs(tokens, pos_tags) do
    pos_seq =
      Enum.map(pos_tags, fn
        {_t, tag} -> normalize_pos(tag)
        tag when is_atom(tag) -> normalize_pos(tag)
        _ -> :X
      end)

    pad =
      case length(tokens) - length(pos_seq) do
        n when n > 0 -> List.duplicate(:X, n)
        _ -> []
      end

    Enum.zip(tokens, pos_seq ++ pad)
  end

  # Maps the analysis's speech-act atom into the small label space the
  # interaction grid expects. Anything we don't recognize falls into
  # `:other`, which is the documented graceful path.
  defp speech_act_label_for_interaction(analysis) do
    sa = get_speech_act(analysis)

    cond do
      Map.get(sa, :is_question) == true -> :question
      sa.sub_type in [:request_information] and sa.category == :directive -> :question
      Map.get(sa, :is_imperative) == true -> :command
      sa.sub_type in [:request_action, :request_information] -> :request
      sa.category == :directive -> :request
      sa.category == :commissive -> :request
      sa.category == :expressive -> :greeting
      sa.category in [:assertive, :declarative] -> :statement
      true -> :other
    end
  end

  # Lowercased surface forms of every content word in the chunk.
  # ConceptNet lookup is case-insensitive at the Lexicon API layer
  # but explicit downcasing here keeps the contract obvious.
  defp content_word_tokens(word_feats) do
    word_feats
    |> Enum.filter(fn wf -> is_map(wf) and Map.get(wf, :is_content_word) == true end)
    |> Enum.map(fn wf -> wf |> Map.get(:token, "") |> to_string() |> String.downcase() end)
    |> Enum.reject(&(&1 == ""))
  end

  @doc """
  Returns the total dimension count of the chunk feature vector.

  Each summand below corresponds to the matching `defp` in this module so
  the declared total stays in lockstep with what `extract/2` actually
  produces. A regression test asserts `length(extract(...)) == vector_dimension()`.
  """
  def vector_dimension do
    # group 1 — surface/lexical
    12 +
      # group 2 — POS distribution (one-hot over @all_pos, 16 tags)
      16 +
      # group 3 — syntactic/structural
      10 +
      # group 4 — pronoun/person/target
      8 +
      # group 5 — modality/certainty
      8 +
      # group 6 — speech act: category one-hot + question-subtype one-hot + 2 flags
      length(@speech_act_categories) + length(@question_subtypes) + 2 +
      # group 7 — discourse: addressee one-hot + 4 indicator flags
      length(@addressee_values) + 4 +
      # group 8 — sentiment
      5 +
      # group 9 — entity: count + density + per-type counts + has_named + new_entity
      2 + length(@entity_types) + 2 +
      # group 10 — lexical-semantic fingerprint over domains
      length(Lexicon.domain_atoms()) +
      # group 11 — word-meaning depth/ambiguity
      6 +
      # group 12 — SRL frames: frame_count + per-role flags + coverage
      1 + length(@srl_roles) + 1 +
      # group 13 — memory/context
      6 +
      # group 14 — slot completeness
      6 +
      # group 15 — wh-target type
      EnrichmentFeatures.wh_dimension() +
      # group 16 — time-expression typology
      EnrichmentFeatures.time_typology_dimension() +
      # group 17 — POS-conditional supersense fingerprints (verb / noun / adj-adv)
      EnrichmentFeatures.verb_supersense_dimension() +
      EnrichmentFeatures.noun_supersense_dimension() +
      EnrichmentFeatures.adj_adv_supersense_dimension() +
      # group 18 — ConceptNet edge-type fingerprint
      EnrichmentFeatures.conceptnet_edge_dimension() +
      # group 19 — verb × argument selectional preferences
      EnrichmentFeatures.selectional_preferences_dimension() +
      # group 20 — subcategorization frame (POS-only)
      EnrichmentFeatures.subcategorization_frame_dimension() +
      # group 21 — discourse-marker semantic categories
      EnrichmentFeatures.discourse_markers_dimension() +
      # group 22 — speech-act × wh-target interaction grid
      EnrichmentFeatures.speech_act_wh_interaction_dimension()
  end

  # -- Group 1: Surface/lexical (~12 dims) ------------------------------------

  defp surface_lexical(tokens, word_feats) do
    token_count = length(tokens)
    char_count = tokens |> Enum.map(&String.length/1) |> Enum.sum()
    avg_word_length = safe_div(char_count, token_count)

    unique_lower = tokens |> Enum.map(&String.downcase/1) |> Enum.uniq() |> length()
    type_token_ratio = safe_div(unique_lower, token_count)

    punct_count = Enum.count(tokens, &punctuation?/1)
    cap_count = Enum.count(tokens, fn t -> t == String.upcase(t) and String.length(t) > 1 end)
    cap_ratio = safe_div(cap_count, token_count)

    contraction_count = Enum.count(tokens, &has_apostrophe?/1)
    has_url = if Enum.any?(tokens, &url?/1), do: 1.0, else: 0.0
    has_quoted = if Enum.any?(tokens, &quoted_start?/1), do: 1.0, else: 0.0

    oov_count = Enum.count(word_feats, & &1.is_oov)
    content_count = Enum.count(word_feats, & &1.is_content_word)
    oov_rate = safe_div(oov_count, max(content_count, 1))

    unique_lemmas =
      word_feats
      |> Enum.filter(& &1.is_content_word)
      |> Enum.map(fn wf -> Lexicon.lemma(wf.token) end)
      |> Enum.uniq()
      |> length()

    [
      min(token_count / 50.0, 1.0),
      min(char_count / 200.0, 1.0),
      min(avg_word_length / 15.0, 1.0),
      type_token_ratio,
      min(punct_count / 10.0, 1.0),
      cap_ratio,
      min(contraction_count / 5.0, 1.0),
      has_url,
      has_quoted,
      oov_rate,
      min(content_count / 20.0, 1.0),
      min(unique_lemmas / 20.0, 1.0)
    ]
  end

  # -- Group 2: POS distribution (~16 dims) -----------------------------------

  defp pos_distribution(pos_tags) do
    total = max(length(pos_tags), 1)

    tag_counts =
      pos_tags
      |> Enum.map(fn
        {_token, tag} -> normalize_pos(tag)
        tag when is_atom(tag) -> normalize_pos(tag)
      end)
      |> Enum.frequencies()

    Enum.map(@all_pos, fn pos ->
      Map.get(tag_counts, pos, 0) / total
    end)
  end

  # -- Group 3: Syntactic/structural (~10 dims) --------------------------------

  defp syntactic_structural(tokens, pos_tags) do
    tags = Enum.map(pos_tags, fn
      {_t, tag} -> normalize_pos(tag)
      tag -> normalize_pos(tag)
    end)

    verb_count = Enum.count(tags, &(&1 == :VERB))
    modal_aux_count = Enum.count(tags, &(&1 == :AUX))
    negation_count = Enum.count(tags, &(&1 == :PART))

    question_mark = if Enum.any?(tokens, &(&1 == "?")), do: 1.0, else: 0.0

    first_tag = List.first(tags)
    imperative_score = if first_tag == :VERB, do: 0.8, else: 0.0
    declarative_score = if first_tag in [:NOUN, :PROPN, :PRON, :DET], do: 0.8, else: 0.2

    comma_count = Enum.count(tokens, &(&1 == ","))
    conj_count = Enum.count(tags, &(&1 == :CONJ))
    clause_depth = min((comma_count + conj_count) / 5.0, 1.0)

    clause_count = max(comma_count + conj_count + 1, 1)
    avg_clause_len = min(length(tokens) / clause_count / 15.0, 1.0)

    subordinate_flag = if conj_count > 0, do: 1.0, else: 0.0

    relative_words = MapSet.new(~w(who whom which that whose))
    relative_flag = if Enum.any?(tokens, fn t -> MapSet.member?(relative_words, String.downcase(t)) end), do: 1.0, else: 0.0

    [
      min(verb_count / 5.0, 1.0),
      min(modal_aux_count / 3.0, 1.0),
      min(negation_count / 3.0, 1.0),
      question_mark,
      imperative_score,
      declarative_score,
      clause_depth,
      avg_clause_len,
      subordinate_flag,
      relative_flag
    ]
  end

  # -- Group 4: Pronoun/person/target (~8 dims) --------------------------------

  defp pronoun_person(tokens) do
    lower_tokens = Enum.map(tokens, &String.downcase/1)

    count_in = fn list ->
      min(Enum.count(lower_tokens, &(&1 in list)) / 3.0, 1.0)
    end

    [
      count_in.(@pronoun_1st_singular),
      count_in.(@pronoun_1st_plural),
      count_in.(@pronoun_2nd),
      count_in.(@pronoun_3rd),
      count_in.(@pronoun_demonstrative),
      count_in.(@pronoun_interrogative),
      count_in.(@pronoun_reflexive),
      count_in.(@pronoun_possessive)
    ]
  end

  # -- Group 5: Modality/certainty (~8 dims) -----------------------------------

  defp modality_certainty(tokens, _pos_tags) do
    lower = Enum.map(tokens, &String.downcase/1)

    can_could = Enum.count(lower, &(&1 in ~w(can could)))
    will_would = Enum.count(lower, &(&1 in ~w(will would)))
    may_might = Enum.count(lower, &(&1 in ~w(may might)))
    should_must = Enum.count(lower, &(&1 in ~w(should must shall ought)))

    conditional = Enum.count(lower, &(&1 in ~w(if unless whether provided assuming)))
    intensifier = Enum.count(lower, &(&1 in ~w(very really extremely absolutely definitely certainly surely)))
    hedge = Enum.count(lower, &(&1 in ~w(maybe perhaps possibly probably likely seemingly apparently kind sort)))
    certainty = Enum.count(lower, &(&1 in ~w(definitely certainly absolutely surely clearly obviously undoubtedly)))

    [
      min(can_could / 2.0, 1.0),
      min(will_would / 2.0, 1.0),
      min(may_might / 2.0, 1.0),
      min(should_must / 2.0, 1.0),
      min(conditional / 2.0, 1.0),
      min(intensifier / 3.0, 1.0),
      min(hedge / 3.0, 1.0),
      min(certainty / 3.0, 1.0)
    ]
  end

  # -- Group 6: Speech act (~10 dims) ------------------------------------------

  defp speech_act_features(analysis) do
    sa = get_speech_act(analysis)

    category_one_hot = Enum.map(@speech_act_categories, fn cat ->
      if sa.category == cat, do: 1.0, else: 0.0
    end)

    question_subtype_one_hot = Enum.map(@question_subtypes, fn st ->
      if sa.sub_type == st, do: 1.0, else: 0.0
    end)

    is_imperative = if sa.is_imperative, do: 1.0, else: 0.0
    is_request = if sa.sub_type in [:request_action, :request_information], do: 1.0, else: 0.0

    category_one_hot ++ question_subtype_one_hot ++ [is_imperative, is_request]
  end

  # -- Group 7: Discourse (~8 dims) --------------------------------------------

  defp discourse_features(analysis) do
    disc = get_discourse(analysis)
    sa = get_speech_act(analysis)

    addressee_one_hot = Enum.map(@addressee_values, fn a ->
      if disc.addressee == a, do: 1.0, else: 0.0
    end)

    disc_indicators = disc.indicators || []
    sa_indicators = Map.get(sa, :indicators) || []
    all_indicators = Enum.map(disc_indicators ++ sa_indicators, &to_string/1)

    discourse_marker_count = min(length(disc_indicators) / 5.0, 1.0)

    greeting =
      if sa.sub_type in [:greeting] or
           sa.category == :expressive and sa.sub_type == :greeting,
         do: 1.0,
         else: 0.0

    farewell =
      if sa.sub_type in [:farewell],
        do: 1.0,
        else: 0.0

    backchannel =
      if sa.sub_type in [:backchannel, :acknowledgment] or
           "backchannel" in all_indicators,
         do: 1.0,
         else: 0.0

    addressee_one_hot ++ [discourse_marker_count, greeting, farewell, backchannel]
  end

  # -- Group 8: Sentiment (~5 dims) --------------------------------------------

  defp sentiment_features(analysis) do
    sent = get_sentiment(analysis)

    label = Map.get(sent, :label) || :neutral
    confidence = Map.get(sent, :confidence) || 0.5

    pos = if label in [:positive, :pos], do: 1.0, else: 0.0
    neg = if label in [:negative, :neg], do: 1.0, else: 0.0
    neu = if label in [:neutral, :neu], do: 1.0, else: 0.0

    polarity_magnitude = abs(pos - neg) * confidence

    [pos, neg, neu, confidence, polarity_magnitude]
  end

  # -- Group 9: Entity (~10 dims) ----------------------------------------------

  defp entity_features(analysis) do
    entities = get_entities(analysis)
    total = length(entities)
    tokens = extract_tokens(analysis)
    token_count = max(length(tokens), 1)

    entity_count = min(total / 10.0, 1.0)
    entity_density = min(total / token_count, 1.0)

    type_counts = Enum.map(@entity_types, fn type ->
      count = Enum.count(entities, fn e ->
        entity_type(e) == type
      end)
      min(count / 3.0, 1.0)
    end)

    has_named = if Enum.any?(entities, fn e -> entity_type(e) in [:person, :location, :organization] end), do: 1.0, else: 0.0

    acc = get_accumulated_context(analysis)
    familiarity = if acc, do: Map.get(acc, :entity_familiarity, 0.5), else: 0.5
    new_entity = if familiarity < 0.3 and total > 0, do: 1.0, else: 0.0

    [entity_count, entity_density] ++ type_counts ++ [has_named, new_entity]
  end

  # -- Group 10: Lexical-semantic fingerprint (~25 dims) -----------------------

  defp lexical_semantic_fingerprint(word_feats) do
    domains = Lexicon.domain_atoms()
    content_words = Enum.filter(word_feats, & &1.is_content_word)
    total = max(length(content_words), 1)

    domain_counts =
      content_words
      |> Enum.map(& &1.lexical_domain)
      |> Enum.reject(&is_nil/1)
      |> Enum.frequencies()

    Enum.map(domains, fn domain ->
      Map.get(domain_counts, domain, 0) / total
    end)
  end

  # -- Group 11: Word-meaning depth/ambiguity (~6 dims) -----------------------

  defp word_meaning_depth(word_feats) do
    content = Enum.filter(word_feats, & &1.is_content_word)

    if content == [] do
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else
      depths = Enum.map(content, & &1.hypernym_depth)
      polys = Enum.map(content, & &1.polysemy_count)

      avg_depth = min(Enum.sum(depths) / max(length(depths), 1) / 20.0, 1.0)
      max_depth = min(Enum.max(depths, fn -> 0 end) / 20.0, 1.0)
      avg_polysemy = min(Enum.sum(polys) / max(length(polys), 1) / 30.0, 1.0)

      min_d = Enum.min(depths, fn -> 0 end)
      max_d = Enum.max(depths, fn -> 0 end)
      abstraction_range = min((max_d - min_d) / 15.0, 1.0)

      avg_similarity = compute_avg_similarity(content)

      antonym_flag =
        if Enum.any?(content, fn wf ->
          case Lexicon.antonyms(wf.token) do
            [] -> false
            _ -> true
          end
        end), do: 1.0, else: 0.0

      [avg_depth, max_depth, avg_polysemy, abstraction_range, avg_similarity, antonym_flag]
    end
  end

  # -- Group 12: Semantic-role frames (~10 dims) --------------------------------

  @propbank_to_thematic %{
    :arg0 => :agent,
    :arg1 => :patient,
    :arg2 => :instrument,
    :arg3 => :recipient,
    :argm_loc => :location,
    :argm_tmp => :time,
    :argm_mnr => :manner,
    :argm_cau => :cause,
    :argm_prp => :cause
  }

  defp srl_frame_features(analysis) do
    frames = get_srl_frames(analysis)
    frame_count = min(length(frames) / 5.0, 1.0)

    all_roles =
      frames
      |> Enum.flat_map(fn
        %{arguments: args} when is_list(args) ->
          Enum.map(args, fn
            %{role: role} -> Map.get(@propbank_to_thematic, role, role)
            _ -> nil
          end)

        _ ->
          []
      end)
      |> Enum.reject(&is_nil/1)
      |> MapSet.new()

    role_flags = Enum.map(@srl_roles, fn role ->
      if MapSet.member?(all_roles, role), do: 1.0, else: 0.0
    end)

    tokens = extract_tokens(analysis)
    coverage = if length(tokens) > 0, do: min(length(frames) / length(tokens), 1.0), else: 0.0

    [frame_count] ++ role_flags ++ [coverage]
  end

  # -- Group 13: Memory/context (~6 dims) --------------------------------------
  #
  # Sourced from the real fields on `Brain.Analysis.ContextAccumulator`:
  #
  #   - `:entity_familiarity`     (0.0..1.0)
  #   - `:relevant_episodes`      (list)
  #   - `:relevant_semantics`     (list)
  #   - `:conversation_topics`    (list)
  #   - `:combined_confidence`    (0.0..1.0)
  #   - `:conflict_measure`       (0.0..1.0)
  #   - `:signals`                (list)
  #
  # Each dim is normalized to 0.0..1.0 so it composes with the rest of the
  # chunk feature vector. When `accumulated_context` is nil (no signals were
  # accumulated for this turn), we fall back to a neutral midpoint vector so
  # downstream classifiers don't read a spurious "perfectly familiar / no
  # conflict" signal.

  @memory_context_default [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]

  defp memory_context_features(analysis) do
    case get_accumulated_context(analysis) do
      nil ->
        @memory_context_default

      acc ->
        familiarity = clamp01(Map.get(acc, :entity_familiarity, 0.5))
        novelty = 1.0 - familiarity

        episode_count = length(Map.get(acc, :relevant_episodes) || [])
        semantic_count = length(Map.get(acc, :relevant_semantics) || [])
        topic_count = length(Map.get(acc, :conversation_topics) || [])
        _signal_count = length(Map.get(acc, :signals) || [])

        similar_episodes = min((episode_count + semantic_count) / 10.0, 1.0)
        graph_known = familiarity
        topic_continuity = min(topic_count / 10.0, 1.0)
        conflict = clamp01(Map.get(acc, :conflict_measure, 0.0))
        combined_conf = clamp01(Map.get(acc, :combined_confidence, 0.5))

        [novelty, similar_episodes, graph_known, topic_continuity, conflict, combined_conf]
    end
  end

  defp clamp01(v) when is_number(v), do: min(max(v * 1.0, 0.0), 1.0)
  defp clamp01(_), do: 0.5

  defp compute_avg_similarity(content_words) when length(content_words) < 2, do: 0.5

  defp compute_avg_similarity(content_words) do
    tokens = Enum.map(content_words, & &1.token)
    pairs = for a <- tokens, b <- tokens, a != b, do: {a, b}

    case pairs do
      [] ->
        0.5

      pairs ->
        sims =
          Enum.map(pairs, fn {a, b} ->
            case Lexicon.word_similarity(a, b) do
              sim when is_number(sim) -> sim
              _ -> 0.0
            end
          end)

        avg = Enum.sum(sims) / max(length(sims), 1)
        min(avg, 1.0)
    end
  rescue
    _ -> 0.5
  end

  # -- Group 14: Slot completeness (~6 dims) -----------------------------------

  defp slot_completeness_features(analysis) do
    slots = get_slots(analysis)

    case slots do
      nil ->
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      %{} = s ->
        filled = map_size(Map.get(s, :filled_slots) || %{})
        missing_req = length(Map.get(s, :missing_required) || [])
        missing_opt = length(Map.get(s, :missing_optional) || [])

        total_req = filled + missing_req

        n_req = min(total_req / 5.0, 1.0)
        n_filled = min(filled / 5.0, 1.0)
        fill_ratio = safe_div(filled, max(total_req, 1))
        has_clarification = if missing_req > 0, do: 1.0, else: 0.0
        missing_count = min(missing_req / 3.0, 1.0)
        filled_optional = max(filled - (total_req - missing_req), 0)
        total_optional = filled_optional + missing_opt
        opt_fill = safe_div(filled_optional, max(total_optional, 1))

        [n_req, n_filled, fill_ratio, has_clarification, missing_count, opt_fill]
    end
  end

  # -- Helpers -----------------------------------------------------------------

  defp extract_tokens(analysis) do
    raw = Map.get(analysis, :text) || ""
    Brain.ML.Tokenizer.tokenize_words(raw)
  rescue
    _ ->
      fallback = Map.get(analysis, :text) || ""
      String.split(to_string(fallback), " ", trim: true)
  end

  defp get_speech_act(analysis) do
    case Map.get(analysis, :speech_act) do
      %{category: _} = sa -> sa
      _ -> %{category: :unknown, sub_type: :unknown, is_question: false, is_imperative: false, indicators: [], confidence: 0.0}
    end
  end

  defp get_discourse(analysis) do
    case Map.get(analysis, :discourse) do
      %{addressee: _} = d -> d
      _ -> %{addressee: :unknown, confidence: 0.0, indicators: []}
    end
  end

  defp get_sentiment(analysis) do
    case Map.get(analysis, :sentiment) do
      %{label: _} = s -> s
      _ -> %{label: :neutral, confidence: 0.5}
    end
  end

  defp get_entities(analysis) do
    Map.get(analysis, :entities) || []
  end

  defp get_srl_frames(analysis) do
    Map.get(analysis, :srl_frames) || []
  end

  defp get_accumulated_context(analysis) do
    Map.get(analysis, :accumulated_context)
  end

  defp get_slots(analysis) do
    Map.get(analysis, :slots)
  end

  defp entity_type(entity) when is_map(entity) do
    type =
      Map.get(entity, :type) ||
        Map.get(entity, "type") ||
        Map.get(entity, :entity_type) ||
        Map.get(entity, "entity_type")

    if is_atom(type), do: type, else: safe_to_atom(type)
  end

  defp entity_type(_), do: :unknown

  defp safe_to_atom(nil), do: :unknown

  defp safe_to_atom(s) when is_binary(s) do
    try do
      String.to_existing_atom(String.downcase(s))
    rescue
      ArgumentError -> :unknown
    end
  end

  defp safe_to_atom(_), do: :unknown

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

  defp punctuation?(token) do
    token
    |> String.graphemes()
    |> Enum.all?(&Brain.ML.Tokenizer.punctuation?/1)
  end

  defp has_apostrophe?(token) do
    String.graphemes(token) |> Enum.member?("'")
  end

  defp quoted_start?(token) do
    String.starts_with?(token, "\"") or String.starts_with?(token, "'")
  end

  defp url?(token), do: String.starts_with?(token, "http") or String.starts_with?(token, "www.")

  defp safe_div(_, denom) when denom == 0, do: 0.0
  defp safe_div(a, b), do: a / b
end
