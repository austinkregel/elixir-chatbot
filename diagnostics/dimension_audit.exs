#!/usr/bin/env elixir
#
# diagnostics/dimension_audit.exs
#
# Performs a dimension-level analysis of the 326-dim feature vector used by
# the FeatureVectorClassifier for intent_domain classification.
#
# Computes:
#   1. Fisher discriminant ratio per dimension (between-class / within-class variance)
#   2. Dead/constant dimension detection
#   3. Per-tier ablation accuracy (classify using only each tier's dims)
#   4. Ranked dimension importance report
#
# Writes a human-readable report to .cursor/notes/feature_dimension_audit.md
#
# Run with:  mix run diagnostics/dimension_audit.exs
#            SAMPLE=5000 mix run diagnostics/dimension_audit.exs  # use more data

sample_size = (System.get_env("SAMPLE") || "5000") |> String.to_integer()

training_path = "data/classifiers/intent_domain.json"

IO.puts("\n╔══════════════════════════════════════════════════════════════╗")
IO.puts("║          DIMENSION AUDIT — 343-dim Feature Vector          ║")
IO.puts("╚══════════════════════════════════════════════════════════════╝")
IO.puts("(If dimension count changed, check group 23 entity type semantics)")

# ── Load training data ────────────────────────────────────────────────

IO.puts("\nLoading training data from #{training_path}…")

raw_entries =
  training_path
  |> File.read!()
  |> Jason.decode!()

training =
  Enum.flat_map(raw_entries, fn
    %{"feature_vector" => v, "label" => l} when is_list(v) and is_binary(l) ->
      [{v, l}]
    _ ->
      []
  end)

IO.puts("Total training examples: #{length(training)}")

dim = training |> List.first() |> elem(0) |> length()
IO.puts("Feature vector dimension: #{dim}")

labels = training |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> Enum.sort()
IO.puts("Classes (#{length(labels)}): #{Enum.join(labels, ", ")}")

label_counts = training |> Enum.map(&elem(&1, 1)) |> Enum.frequencies()
IO.puts("\nClass distribution:")
label_counts
|> Enum.sort_by(fn {_, n} -> -n end)
|> Enum.each(fn {label, n} ->
  pct = Float.round(n / length(training) * 100, 1)
  IO.puts("  #{String.pad_trailing(label, 16)} #{String.pad_leading(Integer.to_string(n), 5)}  (#{pct}%)")
end)

# ── Compute per-dimension statistics ──────────────────────────────────

IO.puts("\nComputing per-dimension Fisher discriminant ratios…")

# Group vectors by class
by_class =
  Enum.group_by(training, fn {_v, l} -> l end, fn {v, _l} -> v end)

# Global mean per dimension
all_vecs = Enum.map(training, &elem(&1, 0))
n_total = length(all_vecs)

global_mean =
  Enum.reduce(all_vecs, List.duplicate(0.0, dim), fn vec, acc ->
    Enum.zip_with(vec, acc, &(&1 + &2))
  end)
  |> Enum.map(&(&1 / n_total))

# Per-class mean and within-class variance
class_stats =
  Map.new(by_class, fn {label, vecs} ->
    n_k = length(vecs)

    class_mean =
      Enum.reduce(vecs, List.duplicate(0.0, dim), fn vec, acc ->
        Enum.zip_with(vec, acc, &(&1 + &2))
      end)
      |> Enum.map(&(&1 / n_k))

    # Within-class variance: (1/n_k) * sum((x_j - mean_j)^2)
    within_var =
      Enum.reduce(vecs, List.duplicate(0.0, dim), fn vec, acc ->
        Enum.zip_with(vec, Enum.zip(class_mean, acc), fn x, {m, a} ->
          a + (x - m) * (x - m)
        end)
      end)
      |> Enum.map(&(&1 / max(n_k, 1)))

    {label, %{mean: class_mean, within_var: within_var, n: n_k}}
  end)

# Between-class variance per dimension: sum_k(n_k * (mean_k_j - global_mean_j)^2) / n_total
between_var =
  Enum.reduce(class_stats, List.duplicate(0.0, dim), fn {_label, stats}, acc ->
    Enum.zip_with(stats.mean, Enum.zip(global_mean, acc), fn m_k, {g, a} ->
      a + stats.n * (m_k - g) * (m_k - g)
    end)
  end)
  |> Enum.map(&(&1 / n_total))

# Within-class variance (pooled): sum_k(n_k * var_k_j) / n_total
within_var_pooled =
  Enum.reduce(class_stats, List.duplicate(0.0, dim), fn {_label, stats}, acc ->
    Enum.zip_with(stats.within_var, acc, fn wv, a ->
      a + stats.n * wv
    end)
  end)
  |> Enum.map(&(&1 / n_total))

# Fisher ratio: between / within (0 if within is 0)
fisher_ratios =
  Enum.zip_with(between_var, within_var_pooled, fn b, w ->
    if w < 1.0e-12, do: 0.0, else: b / w
  end)

# Global variance per dimension (for dead dim detection)
global_var =
  Enum.reduce(all_vecs, List.duplicate(0.0, dim), fn vec, acc ->
    Enum.zip_with(vec, Enum.zip(global_mean, acc), fn x, {m, a} ->
      a + (x - m) * (x - m)
    end)
  end)
  |> Enum.map(&(&1 / n_total))

# ── Dimension name map ────────────────────────────────────────────────

dim_names = %{
  0 => "token_count", 1 => "char_count", 2 => "avg_word_len", 3 => "type_token_ratio",
  4 => "punct_count", 5 => "capitalized_ratio", 6 => "contractions", 7 => "has_url",
  8 => "has_quote", 9 => "oov_rate", 10 => "content_word_count", 11 => "unique_lemma_count",
  12 => "pos_NOUN", 13 => "pos_PROPN", 14 => "pos_VERB", 15 => "pos_AUX",
  16 => "pos_ADJ", 17 => "pos_ADV", 18 => "pos_PRON", 19 => "pos_DET",
  20 => "pos_ADP", 21 => "pos_CONJ", 22 => "pos_PART", 23 => "pos_NUM",
  24 => "pos_INTJ", 25 => "pos_PUNCT", 26 => "pos_SYM", 27 => "pos_X",
  28 => "verb_count", 29 => "modal_aux_count", 30 => "negation_proxy", 31 => "question_mark",
  32 => "imperative_score", 33 => "declarative_start", 34 => "clause_depth", 35 => "avg_clause_len",
  36 => "subordination", 37 => "relative_word", 38 => "pron_1sg", 39 => "pron_1pl",
  40 => "pron_2nd", 41 => "pron_3rd", 42 => "pron_demonstrative", 43 => "pron_interrogative",
  44 => "pron_reflexive", 45 => "pron_possessive", 46 => "modal_can_could", 47 => "modal_will_would",
  48 => "modal_may_might", 49 => "modal_should_must", 50 => "conditional_lexemes", 51 => "intensifiers",
  52 => "hedges", 53 => "certainty_lexemes", 54 => "sa_assertive", 55 => "sa_directive",
  56 => "sa_commissive", 57 => "sa_expressive", 58 => "sa_declarative", 59 => "sa_unknown",
  60 => "q_factual", 61 => "q_opinion", 62 => "q_yes_no", 63 => "is_imperative",
  64 => "is_request", 65 => "addr_bot", 66 => "addr_user", 67 => "addr_other",
  68 => "addr_unknown", 69 => "discourse_indicators", 70 => "greeting_flag", 71 => "farewell_flag",
  72 => "backchannel_flag", 73 => "sent_positive", 74 => "sent_negative", 75 => "sent_neutral",
  76 => "sent_confidence", 77 => "polarity_magnitude", 78 => "entity_count", 79 => "entity_density",
  80 => "ent_person", 81 => "ent_location", 82 => "ent_organization", 83 => "ent_date",
  84 => "ent_number", 85 => "ent_topic", 86 => "ent_device", 87 => "ent_concept",
  88 => "has_named", 89 => "new_entity",
  90 => "ld_adj_all", 91 => "ld_adj_pert", 92 => "ld_adv_all", 93 => "ld_noun_tops",
  94 => "ld_noun_act", 95 => "ld_noun_animal", 96 => "ld_noun_artifact", 97 => "ld_noun_attribute",
  98 => "ld_noun_body", 99 => "ld_noun_cognition", 100 => "ld_noun_communication",
  101 => "ld_noun_event", 102 => "ld_noun_feeling", 103 => "ld_noun_food",
  104 => "ld_noun_group", 105 => "ld_noun_location", 106 => "ld_noun_motive",
  107 => "ld_noun_object", 108 => "ld_noun_person", 109 => "ld_noun_phenomenon",
  110 => "ld_noun_plant", 111 => "ld_noun_possession", 112 => "ld_noun_process",
  113 => "ld_noun_quantity", 114 => "ld_noun_relation", 115 => "ld_noun_shape",
  116 => "ld_noun_state", 117 => "ld_noun_substance", 118 => "ld_noun_time",
  119 => "ld_verb_body", 120 => "ld_verb_change", 121 => "ld_verb_cognition",
  122 => "ld_verb_communication", 123 => "ld_verb_competition", 124 => "ld_verb_consumption",
  125 => "ld_verb_contact", 126 => "ld_verb_creation", 127 => "ld_verb_emotion",
  128 => "ld_verb_motion", 129 => "ld_verb_perception", 130 => "ld_verb_possession",
  131 => "ld_verb_social", 132 => "ld_verb_stative", 133 => "ld_verb_weather",
  134 => "ld_adj_ppl",
  135 => "avg_hypernym_depth", 136 => "max_hypernym_depth", 137 => "avg_polysemy",
  138 => "abstraction_range", 139 => "avg_similarity", 140 => "antonym_present",
  141 => "frame_count", 142 => "role_agent", 143 => "role_patient", 144 => "role_instrument",
  145 => "role_location", 146 => "role_time", 147 => "role_manner", 148 => "role_recipient",
  149 => "role_cause", 150 => "frame_coverage",
  151 => "mem_novelty", 152 => "mem_similar_episodes", 153 => "mem_graph_known",
  154 => "mem_topic_continuity", 155 => "mem_conflict", 156 => "mem_context_signal",
  157 => "slot_required_scale", 158 => "slot_filled_scale", 159 => "slot_fill_ratio",
  160 => "slot_clarification_needed", 161 => "slot_missing_required", 162 => "slot_optional_fill",
  163 => "wh_person", 164 => "wh_entity", 165 => "wh_location", 166 => "wh_time",
  167 => "wh_cause", 168 => "wh_manner",
  169 => "time_deictic", 170 => "time_absolute", 171 => "time_recurring", 172 => "time_duration",
  # Dims 173-217: POS-conditional supersenses (verb 173-187, noun 188-213, adj/adv 214-217)
  173 => "ss_verb_body", 174 => "ss_verb_change", 175 => "ss_verb_cognition",
  176 => "ss_verb_communication", 177 => "ss_verb_competition", 178 => "ss_verb_consumption",
  179 => "ss_verb_contact", 180 => "ss_verb_creation", 181 => "ss_verb_emotion",
  182 => "ss_verb_motion", 183 => "ss_verb_perception", 184 => "ss_verb_possession",
  185 => "ss_verb_social", 186 => "ss_verb_stative", 187 => "ss_verb_weather",
  188 => "ss_noun_act", 189 => "ss_noun_animal", 190 => "ss_noun_artifact",
  191 => "ss_noun_attribute", 192 => "ss_noun_body", 193 => "ss_noun_cognition",
  194 => "ss_noun_communication", 195 => "ss_noun_event", 196 => "ss_noun_feeling",
  197 => "ss_noun_food", 198 => "ss_noun_group", 199 => "ss_noun_location",
  200 => "ss_noun_motive", 201 => "ss_noun_object", 202 => "ss_noun_person",
  203 => "ss_noun_phenomenon", 204 => "ss_noun_plant", 205 => "ss_noun_possession",
  206 => "ss_noun_process", 207 => "ss_noun_quantity", 208 => "ss_noun_relation",
  209 => "ss_noun_shape", 210 => "ss_noun_state", 211 => "ss_noun_substance",
  212 => "ss_noun_time", 213 => "ss_noun_tops",
  214 => "ss_adj_all", 215 => "ss_adj_ppl", 216 => "ss_adj_pert", 217 => "ss_adv_all",
  # Dims 218-229: ConceptNet edge-type fingerprint
  218 => "cn_IsA", 219 => "cn_PartOf", 220 => "cn_HasA", 221 => "cn_UsedFor",
  222 => "cn_CapableOf", 223 => "cn_AtLocation", 224 => "cn_Causes", 225 => "cn_HasProperty",
  226 => "cn_MotivatedByGoal", 227 => "cn_CreatedBy", 228 => "cn_MadeOf", 229 => "cn_ReceivesAction",
  # Dims 230-261: Selectional preferences (32 hash buckets)
  230 => "sp_bucket_0", 231 => "sp_bucket_1", 232 => "sp_bucket_2", 233 => "sp_bucket_3",
  234 => "sp_bucket_4", 235 => "sp_bucket_5", 236 => "sp_bucket_6", 237 => "sp_bucket_7",
  238 => "sp_bucket_8", 239 => "sp_bucket_9", 240 => "sp_bucket_10", 241 => "sp_bucket_11",
  242 => "sp_bucket_12", 243 => "sp_bucket_13", 244 => "sp_bucket_14", 245 => "sp_bucket_15",
  246 => "sp_bucket_16", 247 => "sp_bucket_17", 248 => "sp_bucket_18", 249 => "sp_bucket_19",
  250 => "sp_bucket_20", 251 => "sp_bucket_21", 252 => "sp_bucket_22", 253 => "sp_bucket_23",
  254 => "sp_bucket_24", 255 => "sp_bucket_25", 256 => "sp_bucket_26", 257 => "sp_bucket_27",
  258 => "sp_bucket_28", 259 => "sp_bucket_29", 260 => "sp_bucket_30", 261 => "sp_bucket_31",
  # Dims 262-274: Subcategorization frame
  262 => "subcat_modal_directive", 263 => "subcat_copular", 264 => "subcat_transitive",
  265 => "subcat_ditransitive", 266 => "subcat_intransitive", 267 => "subcat_pron_subject",
  268 => "subcat_verb_norm", 269 => "subcat_noun_norm", 270 => "subcat_aux_norm",
  271 => "subcat_det_norm", 272 => "subcat_adj_norm", 273 => "subcat_adv_norm",
  274 => "subcat_arg_position",
  # Dims 275-283: Discourse-marker categories
  275 => "dm_causal", 276 => "dm_contrast", 277 => "dm_continuation", 278 => "dm_temporal",
  279 => "dm_conditional", 280 => "dm_topic_shift", 281 => "dm_hedge", 282 => "dm_confirmation",
  283 => "dm_marker_density",
  # Dims 284-325: Speech-act x WH interaction grid (6 acts x 7 wh types)
  # Dims 326-342: Entity type semantics (group 23: parent type histogram + coherence + coverage)
  326 => "ets_action", 327 => "ets_artist", 328 => "ets_clothing", 329 => "ets_commerce",
  330 => "ets_communication", 331 => "ets_date_time", 332 => "ets_device",
  333 => "ets_information", 334 => "ets_location", 335 => "ets_measurement",
  336 => "ets_media", 337 => "ets_music_meta", 338 => "ets_number",
  339 => "ets_person", 340 => "ets_weather",
  341 => "ets_coherence", 342 => "ets_coverage",
  284 => "swh_question_who", 285 => "swh_question_what", 286 => "swh_question_where",
  287 => "swh_question_when", 288 => "swh_question_why", 289 => "swh_question_how",
  290 => "swh_question_none", 291 => "swh_request_who", 292 => "swh_request_what",
  293 => "swh_request_where", 294 => "swh_request_when", 295 => "swh_request_why",
  296 => "swh_request_how", 297 => "swh_request_none", 298 => "swh_command_who",
  299 => "swh_command_what", 300 => "swh_command_where", 301 => "swh_command_when",
  302 => "swh_command_why", 303 => "swh_command_how", 304 => "swh_command_none",
  305 => "swh_statement_who", 306 => "swh_statement_what", 307 => "swh_statement_where",
  308 => "swh_statement_when", 309 => "swh_statement_why", 310 => "swh_statement_how",
  311 => "swh_statement_none", 312 => "swh_greeting_who", 313 => "swh_greeting_what",
  314 => "swh_greeting_where", 315 => "swh_greeting_when", 316 => "swh_greeting_why",
  317 => "swh_greeting_how", 318 => "swh_greeting_none", 319 => "swh_other_who",
  320 => "swh_other_what", 321 => "swh_other_where", 322 => "swh_other_when",
  323 => "swh_other_why", 324 => "swh_other_how", 325 => "swh_other_none"
}

dim_name = fn i -> Map.get(dim_names, i, "dim_#{i}") end

# ── Tier definitions ──────────────────────────────────────────────────

tiers = [
  {"Tier 0: Surface/POS/Syntax/Pronoun/Modal/SpeechAct/Discourse/Sent/Entity", 0..89},
  {"Tier 0: Lexical domains + word depth", 90..140},
  {"Tier 0: SRL + Memory + Slots", 141..162},
  {"Tier 1: Enrichments (WH/Time/Supersense/ConceptNet/SelPref)", 163..261},
  {"Tier 2: Enrichments (Subcat/DiscMarker/SpeechActxWH)", 262..325},
  {"Tier 3: Entity Type Semantics (group 23)", 326..342}
]

# ── Report: Dead / constant dimensions ───────────────────────────────

IO.puts("\n\n═══ DEAD / CONSTANT DIMENSIONS ═══")
IO.puts("(global variance < 1e-8)")

dead_dims =
  global_var
  |> Enum.with_index()
  |> Enum.filter(fn {v, _i} -> v < 1.0e-8 end)
  |> Enum.map(fn {v, i} ->
    mean_val = Enum.at(global_mean, i)
    IO.puts("  dim #{String.pad_leading(Integer.to_string(i), 3)} (#{String.pad_trailing(dim_name.(i), 28)}) — constant ≈ #{Float.round(mean_val, 4)}, var=#{:erlang.float_to_binary(v, decimals: 12)}")
    i
  end)

IO.puts("\nTotal dead dims: #{length(dead_dims)} / #{dim}")

# Near-constant (variance < 1e-4 but > 1e-8)
near_constant =
  global_var
  |> Enum.with_index()
  |> Enum.filter(fn {v, _i} -> v >= 1.0e-8 and v < 1.0e-4 end)
  |> Enum.map(fn {v, i} ->
    mean_val = Enum.at(global_mean, i)
    IO.puts("  dim #{String.pad_leading(Integer.to_string(i), 3)} (#{String.pad_trailing(dim_name.(i), 28)}) — near-constant, mean=#{Float.round(mean_val, 4)}, var=#{:erlang.float_to_binary(v, decimals: 8)}")
    i
  end)

IO.puts("Total near-constant dims: #{length(near_constant)} / #{dim}")

# ── Report: Fisher-ranked dimensions ─────────────────────────────────

IO.puts("\n\n═══ FISHER DISCRIMINANT RATIO — TOP 50 ═══")
IO.puts("Higher = better class separation")
IO.puts(String.pad_trailing("rank", 5) <>
  String.pad_trailing("dim", 5) <>
  String.pad_trailing("name", 30) <>
  String.pad_trailing("fisher", 12) <>
  String.pad_trailing("between_var", 14) <>
  "within_var")
IO.puts(String.duplicate("─", 80))

fisher_ranked =
  fisher_ratios
  |> Enum.with_index()
  |> Enum.sort_by(fn {f, _i} -> -f end)

fisher_ranked
|> Enum.take(50)
|> Enum.with_index(1)
|> Enum.each(fn {{f, i}, rank} ->
  IO.puts(
    String.pad_trailing(Integer.to_string(rank), 5) <>
    String.pad_trailing(Integer.to_string(i), 5) <>
    String.pad_trailing(dim_name.(i), 30) <>
    String.pad_trailing(:erlang.float_to_binary(f, decimals: 6), 12) <>
    String.pad_trailing(:erlang.float_to_binary(Enum.at(between_var, i), decimals: 8), 14) <>
    :erlang.float_to_binary(Enum.at(within_var_pooled, i), decimals: 8)
  )
end)

IO.puts("\n═══ FISHER DISCRIMINANT RATIO — BOTTOM 30 ═══")
IO.puts("(least discriminative)")

fisher_ranked
|> Enum.reverse()
|> Enum.take(30)
|> Enum.with_index(dim - 29)
|> Enum.each(fn {{f, i}, rank} ->
  IO.puts(
    String.pad_trailing(Integer.to_string(rank), 5) <>
    String.pad_trailing(Integer.to_string(i), 5) <>
    String.pad_trailing(dim_name.(i), 30) <>
    String.pad_trailing(:erlang.float_to_binary(f, decimals: 6), 12) <>
    String.pad_trailing(:erlang.float_to_binary(Enum.at(between_var, i), decimals: 8), 14) <>
    :erlang.float_to_binary(Enum.at(within_var_pooled, i), decimals: 8)
  )
end)

# ── Per-tier Fisher summary ──────────────────────────────────────────

IO.puts("\n\n═══ PER-TIER FISHER SUMMARY ═══")

Enum.each(tiers, fn {name, range} ->
  tier_fishers = range |> Enum.map(fn i -> Enum.at(fisher_ratios, i) end)
  tier_dim_count = Enum.count(range)
  avg_fisher = Enum.sum(tier_fishers) / max(tier_dim_count, 1)
  max_fisher = Enum.max(tier_fishers)
  min_fisher = Enum.min(tier_fishers)
  nonzero = Enum.count(tier_fishers, &(&1 > 0.001))

  IO.puts("\n  #{name}")
  IO.puts("    dims: #{Enum.at(Enum.to_list(range), 0)}..#{Enum.at(Enum.to_list(range), -1)} (#{tier_dim_count} dims)")
  IO.puts("    avg Fisher: #{Float.round(avg_fisher, 6)}")
  IO.puts("    max Fisher: #{Float.round(max_fisher, 6)}")
  IO.puts("    min Fisher: #{Float.round(min_fisher, 6)}")
  IO.puts("    non-trivial dims (Fisher > 0.001): #{nonzero}/#{tier_dim_count}")
end)

# ── Per-tier ablation accuracy ────────────────────────────────────────

IO.puts("\n\n═══ PER-TIER ABLATION ACCURACY ═══")
IO.puts("(train+classify using only that tier's dims)")

alias Brain.ML.FeatureVectorClassifier

# Shuffle and split for validation
:rand.seed(:exsplus, {42, 137, 256})
shuffled = Enum.shuffle(training)
split_point = div(length(shuffled) * 80, 100)
{train_set, test_set} = Enum.split(shuffled, split_point)

IO.puts("Train/test split: #{length(train_set)} / #{length(test_set)}")

# Full-vector baseline
full_model = FeatureVectorClassifier.train(train_set)
full_correct =
  Enum.count(test_set, fn {vec, label} ->
    case FeatureVectorClassifier.classify(vec, full_model) do
      {:ok, pred, _, _} -> pred == label
      _ -> false
    end
  end)
full_acc = full_correct / max(length(test_set), 1) * 100

IO.puts("\n  FULL (#{dim} dims): #{Float.round(full_acc, 1)}% (#{full_correct}/#{length(test_set)})")

# Per-tier
tier_results =
  Enum.map(tiers, fn {name, range} ->
    indices = Enum.to_list(range)

    tier_train = Enum.map(train_set, fn {vec, label} ->
      tier_vec = Enum.map(indices, fn i -> Enum.at(vec, i) end)
      {tier_vec, label}
    end)

    tier_test = Enum.map(test_set, fn {vec, label} ->
      tier_vec = Enum.map(indices, fn i -> Enum.at(vec, i) end)
      {tier_vec, label}
    end)

    tier_model = FeatureVectorClassifier.train(tier_train)

    correct =
      Enum.count(tier_test, fn {vec, label} ->
        case FeatureVectorClassifier.classify(vec, tier_model) do
          {:ok, pred, _, _} -> pred == label
          _ -> false
        end
      end)

    acc = correct / max(length(tier_test), 1) * 100
    IO.puts("  #{name}: #{Float.round(acc, 1)}% (#{correct}/#{length(tier_test)})")
    {name, acc}
  end)

# Majority-class baseline
majority_label =
  train_set
  |> Enum.map(&elem(&1, 1))
  |> Enum.frequencies()
  |> Enum.max_by(fn {_, n} -> n end)
  |> elem(0)

majority_correct = Enum.count(test_set, fn {_, l} -> l == majority_label end)
majority_acc = majority_correct / max(length(test_set), 1) * 100
IO.puts("\n  Majority-class baseline (\"#{majority_label}\"): #{Float.round(majority_acc, 1)}%")

# ── Cumulative tier combinations ─────────────────────────────────────

IO.puts("\n\n═══ CUMULATIVE TIER ACCURACY ═══")
IO.puts("(adding tiers incrementally)")

cumulative_ranges = [
  {"Tier 0 only (surface through entities)", 0..89},
  {"+ Lexical domains + depth", 0..140},
  {"+ SRL + Memory + Slots", 0..162},
  {"+ Tier 1 enrichments", 0..261},
  {"+ Tier 2 enrichments", 0..325},
  {"+ Tier 3 entity type semantics (FULL)", 0..342}
]

Enum.each(cumulative_ranges, fn {name, range} ->
  indices = Enum.to_list(range)

  cum_train = Enum.map(train_set, fn {vec, label} ->
    cum_vec = Enum.map(indices, fn i -> Enum.at(vec, i) end)
    {cum_vec, label}
  end)

  cum_test = Enum.map(test_set, fn {vec, label} ->
    cum_vec = Enum.map(indices, fn i -> Enum.at(vec, i) end)
    {cum_vec, label}
  end)

  cum_model = FeatureVectorClassifier.train(cum_train)

  correct =
    Enum.count(cum_test, fn {vec, label} ->
      case FeatureVectorClassifier.classify(vec, cum_model) do
        {:ok, pred, _, _} -> pred == label
        _ -> false
      end
    end)

  acc = correct / max(length(cum_test), 1) * 100
  IO.puts("  #{name}: #{Float.round(acc, 1)}%")
end)

# ── Write report to .cursor/notes/ ───────────────────────────────────

report_path = ".cursor/notes/feature_dimension_audit.md"
File.mkdir_p!(Path.dirname(report_path))

report_lines = [
  "# Feature Dimension Audit",
  "",
  "Generated: #{DateTime.utc_now() |> DateTime.to_iso8601()}",
  "Training data: #{training_path} (#{length(training)} examples, #{dim} dims, #{length(labels)} classes)",
  "",
  "## Class Distribution",
  "",
  "| Class | Count | % |",
  "|-------|------:|--:|"
]

class_lines =
  label_counts
  |> Enum.sort_by(fn {_, n} -> -n end)
  |> Enum.map(fn {label, n} ->
    "| #{label} | #{n} | #{Float.round(n / length(training) * 100, 1)}% |"
  end)

dead_section = [
  "",
  "## Dead / Constant Dimensions (#{length(dead_dims)})",
  "",
  "| Dim | Name | Mean | Variance |",
  "|----:|------|-----:|---------:|"
]

dead_lines =
  dead_dims
  |> Enum.map(fn i ->
    "| #{i} | #{dim_name.(i)} | #{Float.round(Enum.at(global_mean, i), 4)} | #{:erlang.float_to_binary(Enum.at(global_var, i), decimals: 12)} |"
  end)

near_const_section = [
  "",
  "## Near-Constant Dimensions (#{length(near_constant)})",
  "",
  "| Dim | Name | Mean | Variance |",
  "|----:|------|-----:|---------:|"
]

near_const_lines =
  near_constant
  |> Enum.map(fn i ->
    "| #{i} | #{dim_name.(i)} | #{Float.round(Enum.at(global_mean, i), 4)} | #{:erlang.float_to_binary(Enum.at(global_var, i), decimals: 8)} |"
  end)

fisher_section = [
  "",
  "## Fisher Discriminant Ratio — Top 50",
  "",
  "| Rank | Dim | Name | Fisher | Between Var | Within Var |",
  "|-----:|----:|------|-------:|------------:|-----------:|"
]

fisher_lines =
  fisher_ranked
  |> Enum.take(50)
  |> Enum.with_index(1)
  |> Enum.map(fn {{f, i}, rank} ->
    "| #{rank} | #{i} | #{dim_name.(i)} | #{Float.round(f, 6)} | #{Float.round(Enum.at(between_var, i), 8)} | #{Float.round(Enum.at(within_var_pooled, i), 8)} |"
  end)

fisher_bottom_section = [
  "",
  "## Fisher Discriminant Ratio — Bottom 30 (least discriminative)",
  "",
  "| Rank | Dim | Name | Fisher | Between Var | Within Var |",
  "|-----:|----:|------|-------:|------------:|-----------:|"
]

fisher_bottom_lines =
  fisher_ranked
  |> Enum.reverse()
  |> Enum.take(30)
  |> Enum.with_index(dim - 29)
  |> Enum.map(fn {{f, i}, rank} ->
    "| #{rank} | #{i} | #{dim_name.(i)} | #{Float.round(f, 6)} | #{Float.round(Enum.at(between_var, i), 8)} | #{Float.round(Enum.at(within_var_pooled, i), 8)} |"
  end)

ablation_section = [
  "",
  "## Ablation Accuracy",
  "",
  "| Configuration | Accuracy |",
  "|---------------|----------|",
  "| FULL (#{dim} dims) | #{Float.round(full_acc, 1)}% |"
]

ablation_lines =
  tier_results
  |> Enum.map(fn {name, acc} -> "| #{name} | #{Float.round(acc, 1)}% |" end)

ablation_lines = ablation_lines ++ ["| Majority-class baseline (\"#{majority_label}\") | #{Float.round(majority_acc, 1)}% |"]

fisher_by_tier_section = [
  "",
  "## Per-Tier Fisher Summary",
  "",
  "| Tier | Dims | Avg Fisher | Max Fisher | Non-trivial |",
  "|------|------|-----------|-----------|-------------|"
]

fisher_by_tier_lines =
  Enum.map(tiers, fn {name, range} ->
    tier_fishers = range |> Enum.map(fn i -> Enum.at(fisher_ratios, i) end)
    tier_dim_count = Enum.count(range)
    avg_fisher = Enum.sum(tier_fishers) / max(tier_dim_count, 1)
    max_fisher = Enum.max(tier_fishers)
    nonzero = Enum.count(tier_fishers, &(&1 > 0.001))
    "| #{name} | #{tier_dim_count} | #{Float.round(avg_fisher, 6)} | #{Float.round(max_fisher, 6)} | #{nonzero}/#{tier_dim_count} |"
  end)

all_report =
  (report_lines ++ class_lines ++ dead_section ++ dead_lines ++
   near_const_section ++ near_const_lines ++
   fisher_section ++ fisher_lines ++
   fisher_bottom_section ++ fisher_bottom_lines ++
   fisher_by_tier_section ++ fisher_by_tier_lines ++
   ablation_section ++ ablation_lines)
  |> Enum.join("\n")

File.write!(report_path, all_report)
IO.puts("\n\nReport written to #{report_path}")
IO.puts("Done.")
