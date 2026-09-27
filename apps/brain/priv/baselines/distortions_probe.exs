# Runs the cognitive-distortions dataset through the pipeline and reports, per axis,
# computed/defaulted and the value distribution. Purpose: test task 081's causal
# claims against 450 sentences that are 97% first-person and 45% negated, rather
# than the 4 hand-written probes 081 relied on.

alias Brain.Analysis.Pipeline
alias Brain.Analysis.FeatureExtractor
alias Brain.Analysis.ChunkProfile

# `mix run` executes from the umbrella root; this matches how train.ex:728 and
# gen_micro_data.ex:579 locate repo-relative data.
csv = Path.join(File.cwd!(), ".cursor/research/cognitive_distortions_dataset.csv")

unless File.exists?(csv) do
  raise """
  cognitive-distortions dataset not found at #{csv}

  This probe measures the axes over first-person, negation-dense input and has
  no meaning without it. There is no fallback corpus.
  """
end

# minimal RFC4180-ish parser (quoted fields may contain commas)
parse_line = fn line ->
  {fields, cur, _in_q} =
    line
    |> String.graphemes()
    |> Enum.reduce({[], "", false}, fn
      "\"", {acc, cur, in_q} -> {acc, cur, not in_q}
      ",", {acc, cur, false} -> {[cur | acc], "", false}
      ch, {acc, cur, in_q} -> {acc, cur <> ch, in_q}
    end)

  Enum.reverse([cur | fields])
end

[_header | lines] = csv |> File.read!() |> String.split("\n", trim: true)
rows = Enum.map(lines, parse_line)

rows =
  Enum.filter(rows, fn r -> length(r) >= 3 and String.trim(Enum.at(r, 0)) != "" end)

IO.puts("parsed #{length(rows)} rows")

negation_re = ~r/\b(not|n't|never|no|nothing|nobody|neither|nor|cannot|can't|won't|don't|didn't|isn't|wasn't)\b/i
firstperson_re = ~r/\b(I|I'm|I've|my|me|myself|I'll|I'd)\b/

axes = ChunkProfile.axes()

results =
  rows
  |> Enum.with_index(1)
  |> Enum.map(fn {[text, _label, dtype | _], i} ->
    if rem(i, 50) == 0, do: IO.puts("  #{i}/#{length(rows)}")
    analysis = Pipeline.analyze_chunk(text)
    vector = FeatureExtractor.extract_vector(analysis)
    p = ChunkProfile.materialize(analysis, vector)

    %{
      text: text,
      dtype: dtype,
      has_negation: Regex.match?(negation_re, text),
      has_first_person: Regex.match?(firstperson_re, text),
      axes: Map.new(axes, fn a -> {a, Map.get(p, a)} end),
      status: Map.new(axes, fn a -> {a, get_in(p.feature_provenance, [a, :status])} end)
    }
  end)

n = length(results)
IO.puts("\n=== per-axis computed / distinct values over #{n} sentences ===\n")
IO.puts(String.pad_trailing("axis", 24) <> String.pad_leading("computed", 10) <> "  values")
IO.puts(String.duplicate("-", 100))

Enum.each(axes, fn axis ->
  computed = Enum.count(results, &(&1.status[axis] == :computed))
  freq = results |> Enum.map(& &1.axes[axis]) |> Enum.frequencies()

  vals =
    freq
    |> Enum.sort_by(&(-elem(&1, 1)))
    |> Enum.map(fn {v, c} -> "#{inspect(v)}=#{c}" end)
    |> Enum.join(", ")

  IO.puts(
    String.pad_trailing(to_string(axis), 24) <>
      String.pad_leading("#{computed}/#{n}", 10) <> "  " <> String.slice(vals, 0, 70)
  )
end)

# --- the two decisive tests -------------------------------------------------
neg = Enum.filter(results, & &1.has_negation)
fp = Enum.filter(results, & &1.has_first_person)

IO.puts("\n=== TEST 1: polarity on #{length(neg)} sentences containing negation tokens ===")
IO.inspect(neg |> Enum.map(& &1.axes[:polarity]) |> Enum.frequencies(), label: "  polarity")
IO.inspect(neg |> Enum.map(& &1.status[:polarity]) |> Enum.frequencies(), label: "  status")

IO.puts("\n=== TEST 2: target/self_disclosure on #{length(fp)} first-person sentences ===")
IO.inspect(fp |> Enum.map(& &1.axes[:target]) |> Enum.frequencies(), label: "  target")
IO.inspect(fp |> Enum.map(& &1.axes[:addressee]) |> Enum.frequencies(), label: "  addressee")
IO.inspect(fp |> Enum.map(& &1.axes[:self_disclosure_level]) |> Enum.frequencies(), label: "  self_disclosure")

IO.puts("\n=== TEST 3: certainty vs absolutist/hedging split ===")
IO.inspect(results |> Enum.map(& &1.axes[:certainty]) |> Enum.frequencies(), label: "  certainty")
IO.inspect(results |> Enum.map(& &1.axes[:speech_act_category]) |> Enum.frequencies(), label: "  speech_act")
