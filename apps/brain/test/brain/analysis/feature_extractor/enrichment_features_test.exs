defmodule Brain.Analysis.FeatureExtractor.EnrichmentFeaturesTest do
  @moduledoc """
  TDD-per-feature suite for the Tier 1 feature enrichments.

  Each `describe` block corresponds to one new feature group. Within a
  block, every test pairs two utterances that should differ on
  *that* group's dimensions and asserts the vectors disagree.

  These tests are written failing-first against
  `Brain.Analysis.FeatureExtractor.EnrichmentFeatures`, a module that
  does not yet exist. They are deliberately phrased without reference
  to specific dimension counts so each enrichment can land
  independently.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.FeatureExtractor.EnrichmentFeatures

  @cn_table :lexicon_conceptnet

  describe "wh_target/1 — the wh-word that frames the question carries domain signal" do
    test "'who' and 'where' produce different vectors (person vs location target)" do
      who_vec = EnrichmentFeatures.wh_target(["who", "is", "calling", "me", "?"])
      where_vec = EnrichmentFeatures.wh_target(["where", "is", "the", "meeting", "?"])

      assert who_vec != where_vec,
             "'who' and 'where' produced identical wh-target vectors. " <>
               "These two interrogatives target completely different semantic " <>
               "categories (person vs. location) and must be separable."
    end

    test "'when' and 'why' produce different vectors (time vs. cause target)" do
      when_vec = EnrichmentFeatures.wh_target(["when", "does", "it", "start", "?"])
      why_vec = EnrichmentFeatures.wh_target(["why", "did", "it", "fail", "?"])

      assert when_vec != why_vec
    end

    test "'how much' is distinct from bare 'how' (quantity vs. manner target)" do
      how_much_vec = EnrichmentFeatures.wh_target(["how", "much", "does", "it", "cost", "?"])
      how_vec = EnrichmentFeatures.wh_target(["how", "do", "i", "fix", "it", "?"])

      assert how_much_vec != how_vec,
             "'how much' (quantity) and bare 'how' (manner) collapsed to the same " <>
               "vector. The quantity reading is a strong signal for account/payment " <>
               "domains and must not be flattened into manner."
    end

    test "a non-question utterance has no wh-target signal" do
      vec = EnrichmentFeatures.wh_target(["please", "open", "the", "garage", "."])

      assert Enum.all?(vec, &(&1 == 0.0)),
             "a sentence without any wh-word lit up at least one wh-target dim: " <>
               inspect(vec)
    end

    test "the wh-target vector is dimension-stable across permutations of non-wh tokens" do
      a = EnrichmentFeatures.wh_target(["who", "is", "austin", "?"])
      b = EnrichmentFeatures.wh_target(["who", "is", "jane", "?"])

      assert a == b,
             "wh-target signal changed when swapping a proper noun. This dimension " <>
               "must be identity-invariant."
    end

    test "wh_dimension/0 declares its size and matches the produced vector" do
      vec = EnrichmentFeatures.wh_target(["who", "is", "there", "?"])
      assert length(vec) == EnrichmentFeatures.wh_dimension()
    end
  end

  describe "time_typology/1 — type of temporal reference shapes intent" do
    test "deictic ('tomorrow') and absolute ('on Monday at 5pm') produce different vectors" do
      deictic = EnrichmentFeatures.time_typology(["see", "you", "tomorrow"])

      absolute =
        EnrichmentFeatures.time_typology(["meet", "on", "monday", "at", "5", "pm"])

      assert deictic != absolute,
             "deictic time reference (tomorrow) and absolute time reference " <>
               "(Monday at 5 pm) collapsed to the same vector — they map to " <>
               "different intent classes (reminder vs calendar)."
    end

    test "recurring ('every day') is distinct from single-shot ('today')" do
      recurring = EnrichmentFeatures.time_typology(["water", "the", "plants", "every", "day"])
      single = EnrichmentFeatures.time_typology(["water", "the", "plants", "today"])

      assert recurring != single,
             "recurring time expression (every day) and single-shot (today) " <>
               "collapsed — recurrence is a strong signal for schedule/timer."
    end

    test "duration ('for 10 minutes') is distinct from a point-in-time" do
      duration = EnrichmentFeatures.time_typology(["run", "the", "timer", "for", "10", "minutes"])
      point = EnrichmentFeatures.time_typology(["start", "the", "timer", "at", "10"])

      assert duration != point,
             "duration ('for 10 minutes') and point-in-time ('at 10') collapsed " <>
               "— duration distinguishes timer from alarm/calendar."
    end

    test "an utterance with no time reference yields all zeros" do
      vec = EnrichmentFeatures.time_typology(["hello", "world"])

      assert Enum.all?(vec, &(&1 == 0.0)),
             "a sentence without any time reference lit up at least one " <>
               "time-typology dim: " <> inspect(vec)
    end

    test "the time-typology vector is identity-invariant under proper-noun substitution" do
      a = EnrichmentFeatures.time_typology(["remind", "austin", "tomorrow"])
      b = EnrichmentFeatures.time_typology(["remind", "jane", "tomorrow"])

      assert a == b,
             "time-typology vector changed when swapping a proper noun. This " <>
               "dimension must be identity-invariant."
    end

    test "time_typology_dimension/0 declares its size and matches the produced vector" do
      vec = EnrichmentFeatures.time_typology(["see", "you", "tomorrow"])
      assert length(vec) == EnrichmentFeatures.time_typology_dimension()
    end
  end

  describe "POS-conditional supersense fingerprints — verb / noun / adj-adv" do
    test "verb_supersenses ignores noun-only word features" do
      noun_only = [
        word_feat(:NOUN, :noun_artifact),
        word_feat(:NOUN, :noun_location)
      ]

      vec = EnrichmentFeatures.verb_supersenses(noun_only)

      assert Enum.all?(vec, &(&1 == 0.0)),
             "verb_supersenses lit up dimensions for a noun-only chunk: " <>
               inspect(vec)
    end

    test "noun_supersenses ignores verb-only word features" do
      verb_only = [
        word_feat(:VERB, :verb_contact),
        word_feat(:VERB, :verb_change)
      ]

      vec = EnrichmentFeatures.noun_supersenses(verb_only)

      assert Enum.all?(vec, &(&1 == 0.0)),
             "noun_supersenses lit up dimensions for a verb-only chunk: " <>
               inspect(vec)
    end

    test "verb_supersenses normalizes over verb count, not total content words" do
      mixed = [
        word_feat(:VERB, :verb_contact),
        word_feat(:NOUN, :noun_artifact),
        word_feat(:NOUN, :noun_location)
      ]

      vec = EnrichmentFeatures.verb_supersenses(mixed)
      verb_total = Enum.sum(vec)

      assert_in_delta verb_total, 1.0, 0.001,
                      "verb_supersenses must sum to 1.0 over the verb subset " <>
                        "(got #{verb_total}). Normalizing over total content words " <>
                        "would bury the verb signal whenever a chunk also has nouns."
    end

    test "noun_supersenses normalizes over noun count, not total content words" do
      mixed = [
        word_feat(:VERB, :verb_contact),
        word_feat(:NOUN, :noun_artifact),
        word_feat(:NOUN, :noun_location)
      ]

      vec = EnrichmentFeatures.noun_supersenses(mixed)
      noun_total = Enum.sum(vec)

      assert_in_delta noun_total, 1.0, 0.001,
                      "noun_supersenses must sum to 1.0 over the noun subset " <>
                        "(got #{noun_total})."
    end

    test "adj_adv_supersenses captures attribute and modifier signal" do
      feats = [
        word_feat(:ADJ, :adj_all),
        word_feat(:ADV, :adv_all)
      ]

      vec = EnrichmentFeatures.adj_adv_supersenses(feats)

      refute Enum.all?(vec, &(&1 == 0.0)),
             "adj_adv_supersenses returned all zeros on an adj+adv chunk: " <>
               inspect(vec)
    end

    test "split-supersense vectors are identity-invariant under domain-preserving substitution" do
      # Two chunks that differ only in token strings but have the same
      # POS+domain fingerprint must produce identical split-supersense
      # vectors. This is the same invariance the bigger test suite
      # asserts at the ChunkProfile level.
      a = [
        word_feat(:VERB, :verb_contact, "turn"),
        word_feat(:NOUN, :noun_artifact, "lights")
      ]

      b = [
        word_feat(:VERB, :verb_contact, "switch"),
        word_feat(:NOUN, :noun_artifact, "thermostat")
      ]

      assert EnrichmentFeatures.verb_supersenses(a) == EnrichmentFeatures.verb_supersenses(b)
      assert EnrichmentFeatures.noun_supersenses(a) == EnrichmentFeatures.noun_supersenses(b)
    end

    test "the three split groups partition Lexicon.domain_atoms/0 with no gap and no overlap" do
      verb_dim = EnrichmentFeatures.verb_supersense_dimension()
      noun_dim = EnrichmentFeatures.noun_supersense_dimension()
      adj_adv_dim = EnrichmentFeatures.adj_adv_supersense_dimension()

      total = verb_dim + noun_dim + adj_adv_dim

      assert total == length(Brain.Lexicon.domain_atoms()),
             "split supersense dims (#{verb_dim} + #{noun_dim} + #{adj_adv_dim} = #{total}) " <>
               "do not partition the lexicon's domain atoms " <>
               "(#{length(Brain.Lexicon.domain_atoms())}). Either a domain is missing " <>
               "from the split or one is double-counted."
    end
  end

  describe "conceptnet_edges/1 — relation-type fingerprint over ConceptNet" do
    setup do
      # Use unique fixture concept names so we don't collide with any
      # real ConceptNet entries already loaded in the ETS table.
      device_concept = "__test_smarthome_device__"
      weather_concept = "__test_weather_phenomenon__"
      music_concept = "__test_music_object__"

      ensure_table()

      # Smarthome-shaped concept: heavy UsedFor + AtLocation + CapableOf.
      :ets.insert(@cn_table, {device_concept, %{
        "UsedFor" => ["light", "illumination", "control"],
        "AtLocation" => ["house", "room", "ceiling"],
        "CapableOf" => ["turn on", "dim", "change color"]
      }})

      # Weather-shaped concept: heavy HasProperty + Causes + IsA.
      :ets.insert(@cn_table, {weather_concept, %{
        "HasProperty" => ["cold", "wet", "windy"],
        "Causes" => ["discomfort", "delay"],
        "IsA" => ["meteorological event"]
      }})

      # Music-shaped concept: heavy IsA + UsedFor.
      :ets.insert(@cn_table, {music_concept, %{
        "IsA" => ["composition", "performance"],
        "UsedFor" => ["entertainment", "expression"],
        "HasProperty" => ["melodic"]
      }})

      on_exit(fn ->
        :ets.delete(@cn_table, device_concept)
        :ets.delete(@cn_table, weather_concept)
        :ets.delete(@cn_table, music_concept)
      end)

      {:ok,
       device: device_concept,
       weather: weather_concept,
       music: music_concept}
    end

    test "smarthome-shaped vs weather-shaped concepts produce different vectors",
         %{device: dev, weather: wx} do
      device_vec = EnrichmentFeatures.conceptnet_edges([dev])
      weather_vec = EnrichmentFeatures.conceptnet_edges([wx])

      assert device_vec != weather_vec,
             "device-shaped and weather-shaped ConceptNet relation profiles " <>
               "collapsed to the same vector — UsedFor/AtLocation/CapableOf " <>
               "cluster (smarthome) is supposed to be separable from " <>
               "HasProperty/Causes (weather)."
    end

    test "an out-of-vocabulary concept yields an all-zero vector" do
      vec = EnrichmentFeatures.conceptnet_edges(["__definitely_not_in_conceptnet_xyz__"])

      assert Enum.all?(vec, &(&1 == 0.0)),
             "an OOV concept produced non-zero ConceptNet edge fingerprint: " <>
               inspect(vec)
    end

    test "the vector is identity-invariant under token reordering",
         %{device: dev, weather: wx} do
      a = EnrichmentFeatures.conceptnet_edges([dev, wx])
      b = EnrichmentFeatures.conceptnet_edges([wx, dev])

      assert a == b,
             "ConceptNet edge fingerprint changed when tokens were reordered. " <>
               "It must be a multiset-style aggregation, not position-sensitive."
    end

    test "conceptnet_edge_dimension/0 declares its size and matches the produced vector",
         %{device: dev} do
      vec = EnrichmentFeatures.conceptnet_edges([dev])
      assert length(vec) == EnrichmentFeatures.conceptnet_edge_dimension()
    end
  end

  describe "selectional_preferences/1 — verb × noun supersense cross-product" do
    test "a chunk with no verbs yields an all-zero vector" do
      noun_only = [word_feat(:NOUN, :noun_artifact)]
      vec = EnrichmentFeatures.selectional_preferences(noun_only)

      assert Enum.all?(vec, &(&1 == 0.0)),
             "selectional_preferences lit up dimensions on a noun-only chunk: " <>
               inspect(vec)
    end

    test "a chunk with no nouns yields an all-zero vector" do
      verb_only = [word_feat(:VERB, :verb_contact)]
      vec = EnrichmentFeatures.selectional_preferences(verb_only)

      assert Enum.all?(vec, &(&1 == 0.0)),
             "selectional_preferences lit up dimensions on a verb-only chunk: " <>
               inspect(vec)
    end

    test "different (verb, noun) supersense pairs land in different buckets" do
      # Pre-compute expected buckets using the same hash + modulus the
      # implementation must use; if both pairs land in the same bucket we
      # skip and pick another pair (this is a guard against a coincidental
      # collision masking a real bug).
      n = EnrichmentFeatures.selectional_preferences_dimension()

      pair_a = {:verb_change, :noun_artifact}
      pair_b = {:verb_communication, :noun_communication}

      bucket_a = :erlang.phash2(pair_a, n)
      bucket_b = :erlang.phash2(pair_b, n)

      assert bucket_a != bucket_b,
             "test fixture collision: pick a different pair. " <>
               inspect({pair_a, bucket_a, pair_b, bucket_b})

      a = [
        word_feat(:VERB, :verb_change),
        word_feat(:NOUN, :noun_artifact)
      ]

      b = [
        word_feat(:VERB, :verb_communication),
        word_feat(:NOUN, :noun_communication)
      ]

      vec_a = EnrichmentFeatures.selectional_preferences(a)
      vec_b = EnrichmentFeatures.selectional_preferences(b)

      assert Enum.at(vec_a, bucket_a) > 0.0,
             "expected bucket #{bucket_a} to have mass for pair #{inspect(pair_a)}, " <>
               "got vector #{inspect(vec_a)}"

      assert Enum.at(vec_b, bucket_b) > 0.0,
             "expected bucket #{bucket_b} to have mass for pair #{inspect(pair_b)}"

      assert vec_a != vec_b
    end

    test "identity invariance under proper-noun substitution" do
      a = [
        word_feat(:VERB, :verb_change, "turn"),
        word_feat(:NOUN, :noun_artifact, "lights")
      ]

      b = [
        word_feat(:VERB, :verb_change, "switch"),
        word_feat(:NOUN, :noun_artifact, "thermostat")
      ]

      assert EnrichmentFeatures.selectional_preferences(a) ==
               EnrichmentFeatures.selectional_preferences(b),
             "selectional_preferences vector changed when the surface tokens were " <>
               "swapped despite the supersense fingerprint being identical."
    end

    test "PROPN does not contribute to noun-side selectional preferences" do
      with_propn = [
        word_feat(:VERB, :verb_communication),
        word_feat(:PROPN, :noun_person, "Austin")
      ]

      with_no_noun = [word_feat(:VERB, :verb_communication)]

      vec_with_propn = EnrichmentFeatures.selectional_preferences(with_propn)
      vec_without = EnrichmentFeatures.selectional_preferences(with_no_noun)

      assert vec_with_propn == vec_without,
             "PROPN contributed to selectional_preferences, which would re-introduce " <>
               "proper-noun identity bias. PROPN must be excluded from the noun side."
    end

    test "selectional_preferences_dimension/0 declares its size and matches output" do
      vec =
        EnrichmentFeatures.selectional_preferences([
          word_feat(:VERB, :verb_contact),
          word_feat(:NOUN, :noun_artifact)
        ])

      assert length(vec) == EnrichmentFeatures.selectional_preferences_dimension()
    end
  end

  describe "subcategorization_frame/1 — verb argument structure shapes intent" do
    test "transitive ('turn the light') and intransitive ('go home') produce different vectors" do
      transitive = EnrichmentFeatures.subcategorization_frame([
        {"turn", :VERB}, {"the", :DET}, {"light", :NOUN}
      ])

      intransitive = EnrichmentFeatures.subcategorization_frame([
        {"go", :VERB}, {"home", :ADV}
      ])

      assert transitive != intransitive,
             "transitive and intransitive frames collapsed to the same vector — " <>
               "argument structure is the strongest predicate-class signal."
    end

    test "copular ('the door is open') is distinct from transitive ('open the door')" do
      copular = EnrichmentFeatures.subcategorization_frame([
        {"the", :DET}, {"door", :NOUN}, {"is", :AUX}, {"open", :ADJ}
      ])

      transitive = EnrichmentFeatures.subcategorization_frame([
        {"open", :VERB}, {"the", :DET}, {"door", :NOUN}
      ])

      assert copular != transitive,
             "copular and transitive frames collapsed — copular skews to " <>
               "smalltalk/statement; transitive skews to action/smarthome."
    end

    test "ditransitive ('send mom a message') is distinct from simple transitive" do
      ditransitive = EnrichmentFeatures.subcategorization_frame([
        {"send", :VERB}, {"mom", :NOUN}, {"a", :DET}, {"message", :NOUN}
      ])

      transitive = EnrichmentFeatures.subcategorization_frame([
        {"send", :VERB}, {"a", :DET}, {"message", :NOUN}
      ])

      assert ditransitive != transitive,
             "ditransitive and transitive collapsed — ditransitive skews to " <>
               "communication/payment domains."
    end

    test "modal-directive ('you can come') is distinct from bare imperative ('come')" do
      modal = EnrichmentFeatures.subcategorization_frame([
        {"you", :PRON}, {"can", :AUX}, {"come", :VERB}, {"tomorrow", :ADV}
      ])

      bare = EnrichmentFeatures.subcategorization_frame([
        {"come", :VERB}, {"tomorrow", :ADV}
      ])

      assert modal != bare,
             "modal-directive and bare imperative collapsed — modality changes " <>
               "directness/politeness mapping for the response posture."
    end

    test "the vector is identity-invariant under noun substitution" do
      a = EnrichmentFeatures.subcategorization_frame([
        {"send", :VERB}, {"mom", :NOUN}, {"a", :DET}, {"message", :NOUN}
      ])

      b = EnrichmentFeatures.subcategorization_frame([
        {"send", :VERB}, {"dad", :NOUN}, {"a", :DET}, {"message", :NOUN}
      ])

      assert a == b,
             "subcategorization vector changed when a noun was swapped — frame " <>
               "extraction must depend only on POS structure."
    end

    test "subcategorization_frame_dimension/0 declares its size and matches the produced vector" do
      vec = EnrichmentFeatures.subcategorization_frame([
        {"turn", :VERB}, {"the", :DET}, {"light", :NOUN}
      ])

      assert length(vec) == EnrichmentFeatures.subcategorization_frame_dimension()
    end
  end

  describe "discourse_markers/1 — connective semantics shape intent" do
    # Discourse markers (PDTB 3.0; Fraser 1999) are a closed-class set
    # of connectives whose category — causal, contrast, conditional,
    # hedge, etc. — signals the rhetorical relation the speaker is
    # building. They sit independently of the propositional content,
    # so they are an ideal identity-invariant feature group: nouns
    # never appear in the marker dictionary.

    test "causal marker ('because') lights up the causal dimension only" do
      vec = EnrichmentFeatures.discourse_markers(["i", "left", "because", "it", "rained"])

      [causal, contrast, continuation, temporal, conditional, topic_shift, hedge, confirmation, _density] =
        vec

      assert causal > 0.0,
             "causal connective 'because' did not raise the causal dim — explanation " <>
               "intent will not separate from neutral statement."

      assert contrast == 0.0
      assert continuation == 0.0
      assert temporal == 0.0
      assert conditional == 0.0
      assert topic_shift == 0.0
      assert hedge == 0.0
      assert confirmation == 0.0
    end

    test "contrast marker ('but') lights up the contrast dimension only" do
      vec = EnrichmentFeatures.discourse_markers(["it", "is", "raining", "but", "warm"])

      [causal, contrast, _cont, _temp, _cond, _ts, _hedge, _conf, _density] = vec

      assert contrast > 0.0,
             "contrast connective 'but' did not raise the contrast dim — correction/objection " <>
               "intents will collapse onto neutral statement."

      assert causal == 0.0
    end

    test "conditional marker ('if') lights up the conditional dimension" do
      vec = EnrichmentFeatures.discourse_markers(["if", "it", "rains", "stay", "home"])

      [_c, _co, _cont, _t, conditional, _ts, _h, _conf, _d] = vec

      assert conditional > 0.0,
             "conditional connective 'if' did not raise the conditional dim — " <>
               "hypothetical / planning intents will not separate from imperative."
    end

    test "hedge marker ('maybe') lights up the hedge dimension" do
      vec = EnrichmentFeatures.discourse_markers(["maybe", "we", "should", "go"])

      [_c, _co, _cont, _t, _cond, _ts, hedge, _conf, _d] = vec

      assert hedge > 0.0,
             "hedge marker 'maybe' did not raise the hedge dim — " <>
               "uncertainty intent will be misclassified as confident assertion."
    end

    test "confirmation marker ('yes') lights up the confirmation dimension" do
      vec = EnrichmentFeatures.discourse_markers(["yes", "please", "do", "it"])

      [_c, _co, _cont, _t, _cond, _ts, _h, confirmation, _d] = vec

      assert confirmation > 0.0,
             "confirmation marker 'yes' did not raise the confirmation dim — " <>
               "acknowledgment intent will collapse onto request intent."
    end

    test "topic-shift marker ('anyway') lights up the topic_shift dimension" do
      vec = EnrichmentFeatures.discourse_markers(["anyway", "let", "us", "move", "on"])

      [_c, _co, _cont, _t, _cond, topic_shift, _h, _conf, _d] = vec

      assert topic_shift > 0.0,
             "topic-shift marker 'anyway' did not raise the topic_shift dim — " <>
               "redirection intent is invisible to the classifier."
    end

    test "multiple markers from different categories all light up (multi-hot)" do
      vec =
        EnrichmentFeatures.discourse_markers([
          "if",
          "it",
          "rains",
          "then",
          "maybe",
          "stay",
          "home"
        ])

      [_c, _co, _cont, temporal, conditional, _ts, hedge, _conf, density] = vec

      assert conditional > 0.0
      assert temporal > 0.0
      assert hedge > 0.0

      assert density > 0.0,
             "marker_density was 0 despite three markers in the chunk — density signal " <>
               "lets the classifier weight chunks built around connectives differently from " <>
               "chunks where a single marker is incidental."
    end

    test "no markers → vector is all zeros" do
      vec = EnrichmentFeatures.discourse_markers(["i", "see", "the", "house"])

      assert Enum.all?(vec, &(&1 == 0.0)),
             "non-zero dimensions emitted for a chunk with zero discourse markers — " <>
               "the marker dictionary is leaking content words."
    end

    test "vector is identity-invariant under noun substitution" do
      a = EnrichmentFeatures.discourse_markers(["because", "austin", "asked"])
      b = EnrichmentFeatures.discourse_markers(["because", "casey", "asked"])

      assert a == b,
             "discourse_markers vector changed when a proper noun was swapped — only " <>
               "closed-class connectives may contribute to this group."
    end

    test "discourse_markers_dimension/0 declares its size and matches the produced vector" do
      vec = EnrichmentFeatures.discourse_markers(["because", "of", "the", "rain"])

      assert length(vec) == EnrichmentFeatures.discourse_markers_dimension()
    end
  end

  describe "speech_act_wh_interaction/2 — explicit cross-product of speech-act × wh-target" do
    # A linear centroid classifier cannot learn feature interactions on
    # its own. Yet (speech_act = QUESTION) × (wh = WHERE) is the most
    # informative joint signal for location/navigation intent; (QUESTION)
    # × (WHEN) is the same for calendar/reminder; (REQUEST) × (HOW) for
    # tutorial/help. Explicitly materializing these interactions as
    # outer-product cells lets the centroid use them additively.

    test "same speech-act with different wh-target produces different vectors" do
      where_q =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["where", "is", "the", "key"])

      when_q =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["when", "is", "the", "meeting"])

      assert where_q != when_q,
             "(question × where) collapsed onto (question × when) — the interaction " <>
               "vector is not selecting on the wh-target dimension."
    end

    test "same wh-target with different speech-act produces different vectors" do
      where_q =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["where", "is", "the", "key"])

      where_req =
        EnrichmentFeatures.speech_act_wh_interaction(:request, ["show", "me", "where", "to", "go"])

      assert where_q != where_req,
             "(question × where) collapsed onto (request × where) — the interaction " <>
               "vector is not selecting on the speech-act dimension."
    end

    test "no wh-token present → the (act, NONE) cell fires and only that cell" do
      vec =
        EnrichmentFeatures.speech_act_wh_interaction(:statement, ["the", "door", "is", "closed"])

      total = Enum.sum(vec)

      assert total == 1.0,
             "expected exactly one cell of mass 1.0 (act × NONE) when no wh-word " <>
               "is present; got total=#{total}. The NONE bucket is missing or " <>
               "non-wh chunks are leaking into wh cells."
    end

    test "unknown speech act maps to the :other row, not a crash" do
      vec =
        EnrichmentFeatures.speech_act_wh_interaction(:nonexistent_act, ["where", "is", "it"])

      assert Enum.sum(vec) == 1.0,
             "unknown speech act did not gracefully fall into the :other row — " <>
               "production would crash on a new act label."
    end

    test "vector is identity-invariant under noun substitution" do
      a =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["where", "is", "austin"])

      b =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["where", "is", "casey"])

      assert a == b,
             "interaction vector changed when a proper noun was swapped — only the " <>
               "speech-act atom and the wh-token set may contribute."
    end

    test "multi-wh chunk multi-hot fires both cells for the same act" do
      vec =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["when", "and", "where", "do", "we", "meet"])

      total = Enum.sum(vec)

      assert total >= 2.0,
             "expected at least two cells (act × when, act × where) to fire for a " <>
               "multi-wh question; got total=#{total}."
    end

    test "speech_act_wh_interaction_dimension/0 declares its size and matches the produced vector" do
      vec =
        EnrichmentFeatures.speech_act_wh_interaction(:question, ["where", "is", "it"])

      assert length(vec) == EnrichmentFeatures.speech_act_wh_interaction_dimension()
    end
  end

  # Ensures the ConceptNet ETS table exists. Normally `Brain.Lexicon.Loader`
  # creates it during application startup, but if a test runs before the
  # loader has finished initializing we create an empty named table here so
  # `:ets.insert/2` doesn't blow up.
  defp ensure_table do
    case :ets.info(@cn_table) do
      :undefined ->
        :ets.new(@cn_table, [:set, :public, :named_table, read_concurrency: true])

      _ ->
        :ok
    end
  end

  # Helper for synthetic word feature maps. Mirrors the shape produced
  # by `Brain.Analysis.FeatureExtractor.WordFeatures.extract/1` but
  # only fills the fields the supersense functions actually read.
  defp word_feat(pos, lexical_domain, token \\ "tok") do
    %{
      token: token,
      pos: pos,
      is_content_word: true,
      features: [],
      lexical_domain: lexical_domain,
      is_oov: false,
      hypernym_depth: 0,
      polysemy_count: 0
    }
  end
end
