# Reads the polarity instrumentation (atom_part vs string_part) that
# `ChunkProfile.derive_polarity/1` was deliberately fitted with, over the
# negation-bearing sentences of the cognitive-distortions set.
#
#     mix run .claude/corpus/polarity_evidence.exs
#
# WHY THIS EXISTS
#
# `chunk_profile.ex` carries a "KNOWN DEFECT, deliberately not fixed here"
# comment and counts both spellings of the PART tag separately, so that two
# independent failures can be told apart in a snapshot:
#
#   * `atom_part` non-zero  -> the atom/string type assumption was right, and a
#                              one-line type fix is all that is needed.
#   * `string_part` non-zero -> the tagger emits PART, so only the type fix
#                              remains (task 074 is then done).
#   * both zero              -> the tagger never emits PART at all, and task 074
#                              is the sole blocker for `polarity`.
#
# Measured 2026-09-12 on 204 negated sentences: both zero. See
# .claude/corpus/baselines/pos_tag_distribution.2026-09-12.baseline.txt.

alias Brain.Analysis.ChunkProfile
alias Brain.Analysis.FeatureExtractor
alias Brain.Analysis.Pipeline

# `mix run` executes from the umbrella root; this matches how train.ex:728 and
# gen_micro_data.ex:579 locate repo-relative data.
csv = Path.join(File.cwd!(), ".cursor/research/cognitive_distortions_dataset.csv")

unless File.exists?(csv) do
  raise """
  cognitive-distortions dataset not found at #{csv}

  This script measures polarity over negation-bearing sentences and has no
  meaning without it. It is not optional and there is no fallback corpus.
  """
end

negation =
  ~r/\b(not|n't|never|no|nothing|nobody|neither|nor|cannot|can't|won't|don't|didn't|isn't|wasn't)\b/i

# The `text` column is the first field. It may be quoted, in which case it can
# contain commas, so a quoted field is read up to its closing quote rather than
# to the first comma. A row that yields no text is a parse failure and is
# reported — never skipped silently.
parse_text = fn
  "\"" <> _ = line ->
    case Regex.run(~r/^"((?:[^"]|"")*)"/, line) do
      [_, text] -> String.replace(text, "\"\"", "\"")
      _ -> :error
    end

  line ->
    case line |> String.split(",", parts: 2) |> List.first() |> String.trim() do
      "" -> :error
      text -> text
    end
end

[_header | lines] = csv |> File.read!() |> String.split("\n", trim: true)

{texts, failures} =
  lines
  |> Enum.map(parse_text)
  |> Enum.split_with(&is_binary/1)

if failures != [] do
  raise "#{length(failures)} row(s) in #{csv} could not be parsed; refusing to report a partial measurement"
end

negated = Enum.filter(texts, &Regex.match?(negation, &1))

IO.puts("parsed #{length(texts)} rows, #{length(negated)} contain a negation token\n")

evidence =
  Enum.map(negated, fn text ->
    analysis = Pipeline.analyze_chunk(text)
    vector = FeatureExtractor.extract_vector(analysis)
    profile = ChunkProfile.materialize(analysis, vector)
    get_in(profile.feature_provenance, [:polarity, :evidence]) || %{}
  end)

sum = fn key -> Enum.reduce(evidence, 0, fn e, acc -> acc + (Map.get(e, key) || 0) end) end

any_part =
  Enum.count(evidence, fn e ->
    (Map.get(e, :atom_part) || 0) + (Map.get(e, :string_part) || 0) > 0
  end)

IO.puts("=== polarity instrumentation over #{length(evidence)} negated sentences ===")
IO.puts("  total pos_tags seen     : #{sum.(:pos_tag_count)}")
IO.puts("  atom_part   (:PART)     : #{sum.(:atom_part)}")
IO.puts("  string_part (\"PART\")    : #{sum.(:string_part)}")
IO.puts("  sentences with any PART : #{any_part}")

IO.puts("""

Reading:
  both zero          -> the tagger emits no PART; task 074 is the sole blocker
  string_part > 0    -> tagger fixed, only the atom/string type fix remains
  atom_part > 0      -> the original type assumption was correct after all
""")
