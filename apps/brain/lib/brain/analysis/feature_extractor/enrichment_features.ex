defmodule Brain.Analysis.FeatureExtractor.EnrichmentFeatures do
  @moduledoc """
  Tier 1 chunk-level feature enrichments for the identity-invariant
  `ChunkProfile` axis classifiers.

  Each function in this module returns a fixed-dimension list of
  floats representing one feature group. The groups are appended to
  the existing 14-group `ChunkFeatures.extract/2` vector by the
  orchestrator so they slot in at the tail without disturbing existing
  offsets.

  ## Why this module exists

  The original 163-dim feature vector captured surface, syntactic,
  speech-act, sentiment, entity, lexical-supersense, depth/polysemy,
  SRL-frame, memory-context, and slot signals. Empirically (gold
  standard regression at 31% domain accuracy after the move to
  feature-vector classifiers) it was missing several signals that the
  literature identifies as primary drivers of intent in short
  utterances:

    * The **wh-word** that frames a question carries strong domain
      signal — *where* skews location/navigation/weather, *when* skews
      time/calendar/reminder, *how much* skews quantity/account.
      (Implemented by `wh_target/1`.)

    * **Time-expression typology** — deictic vs absolute vs recurring
      time references map to different intents (deictic + future ~
      reminder; absolute ~ calendar; recurring ~ schedule/timer).

    * **Verb supersense distinct from noun supersense** — Group 10 of
      the original vector merged WordNet supersenses across POS,
      collapsing `verb.contact` (smarthome) and `noun.artifact`
      (smarthome) into the same fingerprint position as
      `verb.communication` (chat) and `noun.communication` (chat).
      Splitting them lets the centroid distinguish *what kind of
      action* from *what kind of thing*.

    * **ConceptNet relation-type fingerprint** — *UsedFor* +
      *AtLocation* (smarthome devices), *HasProperty* + *Causes*
      (weather phenomena), and *IsA(music)* (music) cluster cleanly
      by domain even when the surface tokens differ.

    * **Verb × argument selectional preference** — the dominant
      explanation signal for intent according to the 2025 NAACL paper
      *"Main Predicate and Their Arguments as Explanation Signals For
      Intent Classification"*. Captures predicate-argument pairs as a
      hashed bigram histogram.

  None of these dimensions can react to specific token identity
  (proper nouns, place names) — they are aggregated through the
  lexicon and through grammatical role, not through string equality.

  ## Research grounding (in `.cursor/research/`)

    * Levin (1993). *English Verb Classes and Alternations.* The
      foundational reference for verb-class as a primary predictor of
      predicate intent.
    * Light & Greiff (2014). *A Comparison of Selectional Preference
      Models for Automatic Verb Classification* (D14-1057). Shows
      verb×argument selectional preferences outperform raw token
      features for verb classification on small data.
    * Tilk et al. (2020). *A neural network approach to selectional
      preference acquisition.* Modern unsupervised approach.
    * Ali et al. (2025). *Main Predicate and Their Arguments as
      Explanation Signals For Intent Classification* (NAACL 2025).
      Establishes verb + argument supersense as the primary
      explanation signal for human-judged intent.
  """

  # ---------------------------------------------------------------------------
  # Group 15: wh-target type
  # ---------------------------------------------------------------------------
  #
  # Six dimensions, one per coarse wh-target category. The mapping
  # follows interrogative pronoun semantics, not surface tokens:
  #
  #   * who/whom/whose      → :person   (target = animate referent)
  #   * what/which          → :entity   (target = thing/property)
  #   * where               → :location (target = place)
  #   * when                → :time     (target = temporal point)
  #   * why                 → :cause    (target = reason)
  #   * how                 → :manner   (target = process), with the
  #                                     bigram "how much"/"how many"
  #                                     promoted to :quantity (slot in
  #                                     :entity for now since it
  #                                     targets a quantity-noun)
  #
  # The output is a count of wh-words in each category, capped at 1.0
  # so multi-clause questions don't blow out the dimension. This mirrors
  # the convention used by groups 4 and 5 in `ChunkFeatures`.

  @wh_categories [:person, :entity, :location, :time, :cause, :manner]

  @wh_lookup %{
    "who" => :person,
    "whom" => :person,
    "whose" => :person,
    "what" => :entity,
    "which" => :entity,
    "where" => :location,
    "when" => :time,
    "why" => :cause,
    "how" => :manner
  }

  @quantity_followers MapSet.new(~w(much many))

  @doc """
  Returns the six-dimension wh-target vector for a list of tokens.

  Order: person, entity, location, time, cause, manner.

  The bigram "how much" / "how many" is reclassified as :entity
  (quantity), not :manner — quantity reading carries account/payment
  domain signal that bare *how* (process/manner) does not.
  """
  @spec wh_target([String.t()]) :: [float()]
  def wh_target(tokens) when is_list(tokens) do
    lower = Enum.map(tokens, &safe_downcase/1)

    counts =
      lower
      |> classify_wh(lower)
      |> Enum.frequencies()

    Enum.map(@wh_categories, fn category ->
      min(Map.get(counts, category, 0) * 1.0, 1.0)
    end)
  end

  @doc "Number of dimensions emitted by `wh_target/1`."
  @spec wh_dimension() :: 6
  def wh_dimension, do: length(@wh_categories)

  # ---- internals ----

  # `classify_wh/2` walks the token list with a lookahead so we can
  # promote "how much"/"how many" out of :manner and into :entity
  # (the quantity reading). Other wh-words classify by the @wh_lookup
  # map; non-wh tokens are skipped.
  defp classify_wh([], _all), do: []

  defp classify_wh([tok | rest], all) do
    case Map.get(@wh_lookup, tok) do
      nil ->
        classify_wh(rest, all)

      :manner ->
        [classify_how(rest) | classify_wh(rest, all)]

      category ->
        [category | classify_wh(rest, all)]
    end
  end

  defp classify_how([next | _]) do
    if MapSet.member?(@quantity_followers, next), do: :entity, else: :manner
  end

  defp classify_how([]), do: :manner

  defp safe_downcase(t) when is_binary(t), do: String.downcase(t)
  defp safe_downcase(other), do: to_string(other)

  # ---------------------------------------------------------------------------
  # Group 16: time-expression typology
  # ---------------------------------------------------------------------------
  #
  # Four dimensions, one per coarse temporal-reference category:
  #
  #   * deictic   — anchored to the moment of utterance:
  #                 "now", "today", "tomorrow", "tonight", "soon"
  #   * absolute  — calendar/clock anchored: weekdays, months,
  #                 time-of-day labels ("noon", "morning", "am/pm")
  #   * recurring — repetition markers: "every", "each", "daily",
  #                 "weekly", "monthly", "yearly", "always"
  #   * duration  — span markers: "minute(s)", "hour(s)", "day(s)",
  #                 "week(s)", "month(s)", "year(s)", "second(s)"
  #
  # Categories are not mutually exclusive on purpose. "every day"
  # lights both `recurring` and `duration`; the centroid can learn
  # that conjunction means *schedule* while `duration` alone means
  # *timer*.
  #
  # All token sets are closed lists drawn from common English
  # temporal vocabulary, in keeping with the project rule against
  # regex-based NLP. Word-class membership is the entire signal.

  @time_categories [:deictic, :absolute, :recurring, :duration]

  @deictic_tokens MapSet.new(
                    ~w(now today tomorrow yesterday tonight later soon currently recently)
                  )

  @recurring_tokens MapSet.new(
                      ~w(every each daily weekly monthly yearly always weekday weekend)
                    )

  @weekday_tokens MapSet.new(
                    ~w(monday tuesday wednesday thursday friday saturday sunday)
                  )

  @month_tokens MapSet.new(
                  ~w(january february march april may june july august september october
                     november december)
                )

  @timeofday_tokens MapSet.new(~w(noon midnight morning afternoon evening night am pm))

  @duration_unit_tokens MapSet.new(
                          ~w(second seconds minute minutes hour hours day days week weeks
                             month months year years)
                        )

  @doc """
  Returns the four-dimension time-typology vector for a list of tokens.

  Order: deictic, absolute, recurring, duration.

  Each dimension is a count of matching tokens, capped at 1.0.
  """
  @spec time_typology([String.t()]) :: [float()]
  def time_typology(tokens) when is_list(tokens) do
    lower = Enum.map(tokens, &safe_downcase/1)

    counts = %{
      deictic: count_in(lower, @deictic_tokens),
      absolute:
        count_in(lower, @weekday_tokens) +
          count_in(lower, @month_tokens) +
          count_in(lower, @timeofday_tokens),
      recurring: count_in(lower, @recurring_tokens),
      duration: count_in(lower, @duration_unit_tokens)
    }

    Enum.map(@time_categories, fn cat ->
      min(Map.get(counts, cat, 0) * 1.0, 1.0)
    end)
  end

  @doc "Number of dimensions emitted by `time_typology/1`."
  @spec time_typology_dimension() :: 4
  def time_typology_dimension, do: length(@time_categories)

  defp count_in(tokens, %MapSet{} = set) do
    Enum.count(tokens, &MapSet.member?(set, &1))
  end

  # ---------------------------------------------------------------------------
  # Group 17: POS-conditional supersense fingerprints
  # ---------------------------------------------------------------------------
  #
  # The original `ChunkFeatures` Group 10 produced a single 45-dim
  # distribution over `Brain.Lexicon.domain_atoms/0` (all WordNet
  # supersenses, regardless of POS). That collapse is information-lossy:
  # a centroid trained on it cannot tell whether the chunk is
  # "predominantly verbs of communication" or "predominantly nouns of
  # communication" because they hit the same dimension positions.
  #
  # We split that single distribution into three POS-conditional
  # ones, each *normalized over its own POS subset* (not the total
  # content-word count). The motivation is straight out of Levin
  # (1993) and the 2025 NAACL "Main Predicate" paper: verb-class is
  # the strongest single predictor of intent, and it deserves its own
  # normalization regime so the verb signal is not diluted whenever a
  # chunk also has nouns.
  #
  # Atoms are grouped by their string prefix (`verb_`, `noun_`,
  # `adj_`/`adv_`). The partition is derived from
  # `Brain.Lexicon.domain_atoms/0` at compile time, so adding a new
  # supersense to the lexicon automatically extends the right group.

  @lexicon_domains Brain.Lexicon.domain_atoms()

  @verb_domains @lexicon_domains
                |> Enum.filter(&String.starts_with?(Atom.to_string(&1), "verb_"))
                |> Enum.sort()

  @noun_domains @lexicon_domains
                |> Enum.filter(&String.starts_with?(Atom.to_string(&1), "noun_"))
                |> Enum.sort()

  @adj_adv_domains @lexicon_domains
                   |> Enum.filter(fn d ->
                     s = Atom.to_string(d)
                     String.starts_with?(s, "adj_") or String.starts_with?(s, "adv_")
                   end)
                   |> Enum.sort()

  @doc """
  Returns the verb-supersense distribution (15 dims) for a list of
  word features.

  Only word features with `pos in [:VERB, :verb]` contribute. The
  vector sums to 1.0 when at least one verb is present and to 0.0
  when none are.
  """
  @spec verb_supersenses([map()]) :: [float()]
  def verb_supersenses(word_feats) when is_list(word_feats) do
    pos_conditional_distribution(word_feats, [:VERB, :verb, "VERB", "verb"], @verb_domains)
  end

  @doc """
  Returns the noun-supersense distribution (26 dims) for a list of
  word features. PROPN is *intentionally excluded* — proper nouns
  carry token identity, which is exactly the bias this whole
  refactor exists to neutralize.
  """
  @spec noun_supersenses([map()]) :: [float()]
  def noun_supersenses(word_feats) when is_list(word_feats) do
    pos_conditional_distribution(word_feats, [:NOUN, :noun, "NOUN", "noun"], @noun_domains)
  end

  @doc """
  Returns the adj+adv supersense distribution (4 dims) for a list of
  word features.
  """
  @spec adj_adv_supersenses([map()]) :: [float()]
  def adj_adv_supersenses(word_feats) when is_list(word_feats) do
    pos_conditional_distribution(
      word_feats,
      [:ADJ, :adj, "ADJ", "adj", :ADV, :adv, "ADV", "adv"],
      @adj_adv_domains
    )
  end

  @doc "Number of dimensions emitted by `verb_supersenses/1`."
  @spec verb_supersense_dimension() :: non_neg_integer()
  def verb_supersense_dimension, do: length(@verb_domains)

  @doc "Number of dimensions emitted by `noun_supersenses/1`."
  @spec noun_supersense_dimension() :: non_neg_integer()
  def noun_supersense_dimension, do: length(@noun_domains)

  @doc "Number of dimensions emitted by `adj_adv_supersenses/1`."
  @spec adj_adv_supersense_dimension() :: non_neg_integer()
  def adj_adv_supersense_dimension, do: length(@adj_adv_domains)

  # ---------------------------------------------------------------------------
  # Group 18: ConceptNet edge-type fingerprint
  # ---------------------------------------------------------------------------
  #
  # Twelve dimensions, one per kept ConceptNet relation type. For each
  # input concept we look up its outgoing edges in
  # `Brain.Lexicon.ConceptNet`, count edges per relation type, and
  # aggregate across all input concepts. The vector is normalized by
  # the total edge count and capped at 1.0 per dim.
  #
  # The intuition: domain semantics imprint on the *shape* of a
  # concept's relation profile. A smarthome device has many
  # `UsedFor`/`AtLocation`/`CapableOf` edges. A weather phenomenon
  # has many `HasProperty`/`Causes` edges. A piece of music has many
  # `IsA`/`UsedFor(entertainment)` edges. The exact related concepts
  # don't matter for this feature — only the *type distribution*.
  # That's the property that makes this dim identity-invariant: even
  # if we swap "lights" for "thermostat", both concepts have the same
  # relation-type signature.
  #
  # Selection rationale: of the ~21 relation types ConceptNetParser
  # ingests, we keep 12 with the strongest domain-discriminative
  # power. We deliberately exclude the long tail (HasFirstSubevent,
  # SymbolOf, etc.) because they are sparse in ConceptNet 5 and add
  # noise.

  @conceptnet_edge_types ~w(IsA PartOf HasA UsedFor CapableOf AtLocation
                            Causes HasProperty MotivatedByGoal CreatedBy
                            MadeOf ReceivesAction)

  @doc """
  Returns the twelve-dimension ConceptNet edge-type fingerprint for a
  list of concept tokens.

  Tokens are looked up in `Brain.Lexicon.ConceptNet` (case-insensitive
  via the underlying lookup). OOV tokens contribute zero edges.
  """
  @spec conceptnet_edges([String.t()]) :: [float()]
  def conceptnet_edges(concepts) when is_list(concepts) do
    aggregated =
      Enum.reduce(concepts, %{}, fn concept, acc ->
        case Brain.Lexicon.ConceptNet.relation_counts(to_string(concept)) do
          rels when is_map(rels) ->
            Enum.reduce(rels, acc, fn {rel_type, count}, inner ->
              Map.update(inner, rel_type, count, &(&1 + count))
            end)

          _ ->
            acc
        end
      end)

    total = aggregated |> Map.values() |> Enum.sum() |> max(1)

    Enum.map(@conceptnet_edge_types, fn rel ->
      raw = Map.get(aggregated, rel, 0)
      min(raw / total, 1.0)
    end)
  end

  @doc "Number of dimensions emitted by `conceptnet_edges/1`."
  @spec conceptnet_edge_dimension() :: 12
  def conceptnet_edge_dimension, do: length(@conceptnet_edge_types)

  # ---------------------------------------------------------------------------
  # Group 19: verb × argument selectional-preference cross-product
  # ---------------------------------------------------------------------------
  #
  # Hashed cross-product of (verb_supersense, noun_supersense) pairs
  # for every (VERB, NOUN) combination in the chunk. We hash to a
  # fixed bucket count because a dense cross-product (15 verb supersenses
  # × 26 noun supersenses = 390 dims) is too sparse for our centroid
  # classifier to learn from on a 5,300-example training set.
  #
  # Why this matters: per Levin (1993) and the 2025 NAACL "Main Predicate
  # and Their Arguments" paper, the dominant explanation signal for
  # short-utterance intent is the conjunction of predicate class and
  # argument class — not either alone. "verb of contact" is ambiguous
  # (smarthome? sports?); "verb of contact + artifact noun" is
  # smarthome. The split-supersense group (Group 17) gives the centroid
  # the marginals; this group gives it the joint.
  #
  # PROPN is *intentionally excluded* from the noun side. Otherwise
  # "call <PERSON>" and "remind <PERSON>" would all collapse onto one
  # bucket through (verb_communication, noun_person), and the proper-
  # noun bias we just neutralized would creep back through this
  # feature's back door.

  @selpref_buckets 32

  @doc """
  Returns the 32-dimension selectional-preference fingerprint for a
  list of word features.

  Pairs every (VERB, NOUN) word-feature combination, hashes the
  `{verb_supersense, noun_supersense}` tuple to one of
  `selectional_preferences_dimension/0` buckets via `:erlang.phash2/2`,
  and returns the normalized bucket counts (capped at 1.0).

  PROPN is excluded from the noun side to preserve identity invariance.
  """
  @spec selectional_preferences([map()]) :: [float()]
  def selectional_preferences(word_feats) when is_list(word_feats) do
    verbs = filter_with_domain(word_feats, [:VERB, :verb, "VERB", "verb"])
    nouns = filter_with_domain(word_feats, [:NOUN, :noun, "NOUN", "noun"])

    case {verbs, nouns} do
      {[], _} -> List.duplicate(0.0, @selpref_buckets)
      {_, []} -> List.duplicate(0.0, @selpref_buckets)
      {vs, ns} -> hash_pairs(vs, ns)
    end
  end

  @doc "Number of buckets / dimensions emitted by `selectional_preferences/1`."
  @spec selectional_preferences_dimension() :: 32
  def selectional_preferences_dimension, do: @selpref_buckets

  defp filter_with_domain(word_feats, allowed_pos) do
    pos_set = MapSet.new(allowed_pos)

    word_feats
    |> Enum.filter(fn wf ->
      is_map(wf) and Map.get(wf, :is_content_word) == true and
        MapSet.member?(pos_set, Map.get(wf, :pos)) and
        not is_nil(Map.get(wf, :lexical_domain))
    end)
    |> Enum.map(& &1.lexical_domain)
  end

  defp hash_pairs(verb_supers, noun_supers) do
    pairs =
      for v <- verb_supers, n <- noun_supers do
        :erlang.phash2({v, n}, @selpref_buckets)
      end

    total = max(length(pairs), 1)
    counts = Enum.frequencies(pairs)

    Enum.map(0..(@selpref_buckets - 1), fn bucket ->
      min(Map.get(counts, bucket, 0) / total, 1.0)
    end)
  end

  defp pos_conditional_distribution(word_feats, allowed_pos, domains) do
    pos_set = MapSet.new(allowed_pos)

    filtered =
      Enum.filter(word_feats, fn wf ->
        is_map(wf) and Map.get(wf, :is_content_word) == true and
          MapSet.member?(pos_set, Map.get(wf, :pos)) and
          not is_nil(Map.get(wf, :lexical_domain))
      end)

    case filtered do
      [] ->
        List.duplicate(0.0, length(domains))

      _ ->
        total = length(filtered)

        counts =
          filtered
          |> Enum.map(& &1.lexical_domain)
          |> Enum.frequencies()

        Enum.map(domains, fn d -> Map.get(counts, d, 0) / total end)
    end
  end

  # ──────────────────────────────────────────────────────────────────────
  # Tier 2 / Feature 6 — subcategorization frame
  # ──────────────────────────────────────────────────────────────────────
  #
  # Verb argument structure (Levin 1993; Light & Greiff 2014) is one of
  # the strongest predicate-class signals for intent. *Transitive*
  # ("turn the light"), *intransitive* ("go home"), *copular* ("the door
  # is open"), *ditransitive* ("send mom a message") and *modal-directive*
  # ("you can come tomorrow") map to systematically different downstream
  # response postures even when they share head verbs.
  #
  # This group operates **only on the POS sequence**, never on tokens.
  # That is the strict identity-invariance contract: swapping any noun
  # for any other noun must leave the vector unchanged. The function
  # accepts `[{token, pos}]` for ergonomic call-sites but discards
  # tokens immediately.
  #
  # Output dimensions (13):
  #
  #   0  has_modal_directive   — AUX immediately followed by VERB
  #   1  has_copular           — AUX immediately followed by ADJ/NOUN/PROPN
  #   2  has_transitive        — first VERB followed by exactly one NP
  #   3  has_ditransitive      — first VERB followed by ≥2 NPs
  #   4  has_intransitive      — first VERB followed by zero NPs
  #   5  has_pron_subject      — PRON precedes the first VERB/AUX
  #   6  verb_count_norm       — VERB count / chunk length
  #   7  noun_count_norm       — NOUN+PROPN count / chunk length
  #   8  aux_count_norm        — AUX count / chunk length
  #   9  det_count_norm        — DET count / chunk length
  #  10  adj_count_norm        — ADJ count / chunk length
  #  11  adv_count_norm        — ADV count / chunk length
  #  12  arg_position_pattern  — categorical chunk-initial constituent
  #
  # NPs are detected as `(DET? (NOUN | PROPN | PRON))` heads. We do not
  # distinguish PRON from NOUN here for object-counting purposes
  # because both behave as direct objects in subcategorization frames.

  @subcat_dim 13

  @doc """
  Returns the 13-dimension subcategorization-frame vector for a chunk
  given its `[{token, pos}]` token sequence.

  Reads only the POS dimension of each tuple — token identity is
  intentionally discarded so the output is invariant under noun
  substitution.
  """
  @spec subcategorization_frame([{String.t(), atom()}]) :: [float()]
  def subcategorization_frame([]), do: List.duplicate(0.0, @subcat_dim)

  def subcategorization_frame(tokens_pos) when is_list(tokens_pos) do
    pos_seq =
      Enum.map(tokens_pos, fn
        {_tok, pos} -> normalize_pos_tag(pos)
        _ -> :UNKNOWN
      end)

    pairs = Enum.zip(pos_seq, tl(pos_seq) ++ [nil])

    has_modal_directive =
      Enum.any?(pairs, fn {a, b} -> a == :AUX and b == :VERB end)

    has_copular =
      Enum.any?(pairs, fn {a, b} ->
        a == :AUX and b in [:ADJ, :NOUN, :PROPN]
      end)

    {has_transitive, has_ditransitive, has_intransitive} =
      case Enum.find_index(pos_seq, &(&1 == :VERB)) do
        nil ->
          {false, false, false}

        idx ->
          rest = Enum.drop(pos_seq, idx + 1)

          case count_object_nps(rest) do
            0 -> {false, false, true}
            1 -> {true, false, false}
            _ -> {false, true, false}
          end
      end

    has_pron_subject =
      case Enum.find_index(pos_seq, &(&1 in [:VERB, :AUX])) do
        nil -> false
        idx -> Enum.any?(Enum.take(pos_seq, idx), &(&1 == :PRON))
      end

    total = max(length(pos_seq), 1)
    norm = fn count -> count / total end

    verb_count = Enum.count(pos_seq, &(&1 == :VERB))
    noun_count = Enum.count(pos_seq, fn p -> p in [:NOUN, :PROPN] end)
    aux_count = Enum.count(pos_seq, &(&1 == :AUX))
    det_count = Enum.count(pos_seq, &(&1 == :DET))
    adj_count = Enum.count(pos_seq, &(&1 == :ADJ))
    adv_count = Enum.count(pos_seq, &(&1 == :ADV))

    arg_position =
      case List.first(pos_seq) do
        :VERB -> 0.0
        p when p in [:NOUN, :PROPN, :PRON, :DET] -> 0.5
        _ -> 1.0
      end

    [
      bool_to_float(has_modal_directive),
      bool_to_float(has_copular),
      bool_to_float(has_transitive),
      bool_to_float(has_ditransitive),
      bool_to_float(has_intransitive),
      bool_to_float(has_pron_subject),
      norm.(verb_count),
      norm.(noun_count),
      norm.(aux_count),
      norm.(det_count),
      norm.(adj_count),
      norm.(adv_count),
      arg_position
    ]
  end

  @doc "Number of dimensions emitted by `subcategorization_frame/1`."
  @spec subcategorization_frame_dimension() :: 13
  def subcategorization_frame_dimension, do: @subcat_dim

  # Counts object-position NPs in a POS-suffix. An NP head is any of
  # NOUN / PROPN / PRON, optionally preceded by a DET. We walk the
  # sequence consuming one head per NP so adjacent NPs (as in the
  # ditransitive "send mom a message" → [NOUN, DET, NOUN]) are counted
  # independently.
  defp count_object_nps(pos_seq), do: do_count_nps(pos_seq, 0)

  defp do_count_nps([], acc), do: acc

  defp do_count_nps([:DET, head | tail], acc)
       when head in [:NOUN, :PROPN, :PRON] do
    do_count_nps(tail, acc + 1)
  end

  defp do_count_nps([head | tail], acc)
       when head in [:NOUN, :PROPN, :PRON] do
    do_count_nps(tail, acc + 1)
  end

  defp do_count_nps([_ | tail], acc), do: do_count_nps(tail, acc)

  defp bool_to_float(true), do: 1.0
  defp bool_to_float(false), do: 0.0

  # ──────────────────────────────────────────────────────────────────────
  # Tier 2 / Feature 7 — discourse-marker semantic categories
  # ──────────────────────────────────────────────────────────────────────
  #
  # Discourse markers (PDTB 3.0 Webber et al. 2019; Fraser 1999) are a
  # closed-class lexical set of connectives that signal the rhetorical
  # relation a speaker is constructing — *causal*, *contrast*,
  # *conditional*, *hedge*, etc. They are propositional-content
  # independent: substituting any noun in the chunk leaves the marker
  # vector unchanged because nouns can never appear in the marker
  # dictionary.
  #
  # Eight categories cover the relations that most differentiate
  # response posture for a chat-style assistant:
  #
  #   * causal       — "because", "so", "therefore" → explanation
  #   * contrast     — "but", "however"             → correction / objection
  #   * continuation — "and", "also", "moreover"    → list / elaboration
  #   * temporal     — "then", "now", "while"       → sequencing / scheduling
  #   * conditional  — "if", "unless", "otherwise"  → hypothetical / planning
  #   * topic_shift  — "anyway", "incidentally"     → redirection
  #   * hedge        — "maybe", "perhaps"           → uncertainty / mitigation
  #   * confirmation — "yes", "ok", "sure"          → acknowledgment / dialog continuation
  #
  # We deliberately omit modal AUX ("can", "should", "must") — those are
  # already captured by the subcategorization-frame group's
  # `has_modal_directive` and `aux_count_norm` dimensions, and including
  # them here would create redundant covariance.
  #
  # Output dimensions (9):
  #
  #   0..7  per-category presence — count_in_category / total_tokens, capped at 1.0
  #   8     density — total_markers_found / total_tokens, capped at 1.0
  #
  # Density lets the classifier weight chunks that are *built around*
  # connectives differently from chunks where a single marker is
  # incidental.

  @dm_categories %{
    causal: ~w(because since so therefore thus hence consequently accordingly),
    contrast: ~w(but however although though yet whereas nevertheless nonetheless),
    continuation: ~w(and also moreover furthermore additionally besides plus),
    temporal: ~w(then now after before while when meanwhile eventually finally later),
    conditional: ~w(if unless otherwise),
    topic_shift: ~w(anyway incidentally regardless),
    hedge: ~w(maybe perhaps possibly probably arguably likely apparently),
    confirmation: ~w(yes yeah yep ok okay sure right exactly absolutely definitely)
  }

  @dm_order [
    :causal,
    :contrast,
    :continuation,
    :temporal,
    :conditional,
    :topic_shift,
    :hedge,
    :confirmation
  ]

  @dm_token_to_category (for {cat, toks} <- @dm_categories,
                             t <- toks,
                             into: %{},
                             do: {t, cat})

  @dm_marker_set MapSet.new(Map.keys(@dm_token_to_category))

  @discmark_dim length(@dm_order) + 1

  @doc """
  Returns the 9-dimension discourse-marker vector for a chunk's
  list of token strings (already tokenized, case-insensitive).

  The first eight dims are per-category normalized counts (over total
  tokens, capped at 1.0); the ninth is the overall marker density.

  Identity-invariant by construction: nouns and proper nouns are not
  in the marker dictionary, so swapping them leaves the vector
  unchanged.
  """
  @spec discourse_markers([String.t()]) :: [float()]
  def discourse_markers(tokens) when is_list(tokens) do
    total = max(length(tokens), 1)

    hits =
      tokens
      |> Enum.map(&String.downcase/1)
      |> Enum.filter(&MapSet.member?(@dm_marker_set, &1))
      |> Enum.map(&Map.fetch!(@dm_token_to_category, &1))

    counts = Enum.frequencies(hits)

    per_category =
      Enum.map(@dm_order, fn cat ->
        min(Map.get(counts, cat, 0) / total, 1.0)
      end)

    density = min(length(hits) / total, 1.0)

    per_category ++ [density]
  end

  @doc "Number of dimensions emitted by `discourse_markers/1`."
  @spec discourse_markers_dimension() :: 9
  def discourse_markers_dimension, do: @discmark_dim

  # ──────────────────────────────────────────────────────────────────────
  # Tier 2 / Feature 8 — speech-act × wh-target interaction grid
  # ──────────────────────────────────────────────────────────────────────
  #
  # A linear centroid classifier sums per-feature contributions; it
  # cannot represent joint signals like "this is a *question* AND it
  # asks about *where*". Yet that joint is the strongest single
  # predictor of the location/navigation domain, and analogous joints
  # drive calendar (question × when), tutorial (request × how), and
  # smalltalk (statement × none).
  #
  # We materialize the interaction explicitly as a 6 × 7 = 42-cell
  # one-hot/multi-hot grid. Cells are addressable as
  # `index = act_idx * length(@swh_wh) + wh_idx`.
  #
  #   speech-act rows (6) — :question, :request, :command, :statement,
  #                         :greeting, :other
  #   wh-target cols (7)  — :who, :what, :where, :when, :why, :how, :none
  #
  # Identity invariance: only the speech-act atom and the lowercased
  # wh-token *type* contribute. No content-bearing token (noun, verb,
  # proper noun) ever influences the output.

  @swh_acts [:question, :request, :command, :statement, :greeting, :other]
  @swh_wh [:who, :what, :where, :when, :why, :how, :none]

  @swh_dim length(@swh_acts) * length(@swh_wh)

  @swh_wh_token_to_idx %{
    "who" => 0,
    "what" => 1,
    "where" => 2,
    "when" => 3,
    "why" => 4,
    "how" => 5
  }

  @swh_wh_token_set MapSet.new(Map.keys(@swh_wh_token_to_idx))

  @doc """
  Returns the 42-dimension `(speech-act × wh-target)` interaction grid
  for a chunk.

  The grid is a flat row-major matrix of `length(@swh_acts) ×
  length(@swh_wh)` cells. The selected speech-act row is multi-hot
  across the wh-target columns (one column per wh-word actually
  present, or the `:none` column when the chunk has no wh-token).

  Identity-invariant by construction.
  """
  @spec speech_act_wh_interaction(atom(), [String.t()]) :: [float()]
  def speech_act_wh_interaction(speech_act, tokens)
      when is_atom(speech_act) and is_list(tokens) do
    act_idx = swh_act_index(speech_act)
    wh_indices = swh_wh_indices(tokens)
    n_cols = length(@swh_wh)

    set_indices = MapSet.new(Enum.map(wh_indices, fn col -> act_idx * n_cols + col end))

    Enum.map(0..(@swh_dim - 1), fn i ->
      if MapSet.member?(set_indices, i), do: 1.0, else: 0.0
    end)
  end

  @doc "Number of dimensions emitted by `speech_act_wh_interaction/2`."
  @spec speech_act_wh_interaction_dimension() :: 42
  def speech_act_wh_interaction_dimension, do: @swh_dim

  defp swh_act_index(act) do
    case Enum.find_index(@swh_acts, &(&1 == act)) do
      nil -> length(@swh_acts) - 1
      idx -> idx
    end
  end

  # Returns the column indices that should fire for this token list.
  # Multi-hot if multiple wh-words appear; collapses to the single
  # `:none` column index when no wh-word is present.
  defp swh_wh_indices(tokens) do
    cols =
      tokens
      |> Enum.map(&String.downcase/1)
      |> Enum.filter(&MapSet.member?(@swh_wh_token_set, &1))
      |> Enum.map(&Map.fetch!(@swh_wh_token_to_idx, &1))
      |> Enum.uniq()

    case cols do
      [] -> [length(@swh_wh) - 1]
      _ -> cols
    end
  end

  defp normalize_pos_tag(pos) when pos in [:NOUN, "NOUN", "noun"], do: :NOUN
  defp normalize_pos_tag(pos) when pos in [:PROPN, "PROPN", "propn"], do: :PROPN
  defp normalize_pos_tag(pos) when pos in [:VERB, "VERB", "verb"], do: :VERB
  defp normalize_pos_tag(pos) when pos in [:ADJ, "ADJ", "adj"], do: :ADJ
  defp normalize_pos_tag(pos) when pos in [:ADV, "ADV", "adv"], do: :ADV
  defp normalize_pos_tag(pos) when pos in [:AUX, "AUX", "aux"], do: :AUX
  defp normalize_pos_tag(pos) when pos in [:PRON, "PRON", "pron"], do: :PRON
  defp normalize_pos_tag(pos) when pos in [:DET, "DET", "det"], do: :DET
  defp normalize_pos_tag(pos) when pos in [:ADP, "ADP", "adp"], do: :ADP
  defp normalize_pos_tag(pos) when pos in [:CONJ, :CCONJ, :SCONJ, "CONJ", "CCONJ", "SCONJ"], do: :CONJ
  defp normalize_pos_tag(pos) when pos in [:PART, "PART", "part"], do: :PART
  defp normalize_pos_tag(pos) when pos in [:NUM, "NUM", "num"], do: :NUM
  defp normalize_pos_tag(pos) when pos in [:INTJ, "INTJ", "intj"], do: :INTJ
  defp normalize_pos_tag(pos) when pos in [:PUNCT, "PUNCT", "punct"], do: :PUNCT
  defp normalize_pos_tag(pos) when pos in [:SYM, "SYM", "sym"], do: :SYM
  defp normalize_pos_tag(pos) when is_atom(pos), do: pos
  defp normalize_pos_tag(_), do: :UNKNOWN
end
