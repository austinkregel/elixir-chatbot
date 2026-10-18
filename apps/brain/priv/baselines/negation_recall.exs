alias Brain.LinguisticData

probes = [
  "I do not like this",
  "I don't want that",
  "I can't help it",
  "I never do anything right.",
  "There's no point in trying",
  "nobody loves me",
  "Nothing will change that.",
  "I won't be able to sleep tonight.",
  "I like rain"
]

IO.puts("=== LinguisticData.has_negation?/1 on hand probes ===")

Enum.each(probes, fn t ->
  IO.puts("  #{String.pad_trailing(inspect(t), 42)} -> #{LinguisticData.has_negation?(t)}")
end)

IO.puts("\n=== tokenisation used by has_negation?/1 (contractions expanded first) ===")

Enum.each(["I don't want that", "I can't help it", "I won't sleep"], fn t ->
  expanded = Brain.ML.Tokenizer.expand_contractions(t)
  toks = Brain.ML.Tokenizer.tokenize_normalized(expanded, expand_contractions: false)
  IO.puts("  #{String.pad_trailing(inspect(t), 22)} -> #{String.pad_trailing(inspect(expanded), 24)} #{inspect(toks)}")
end)

# `mix run` executes from the umbrella root; this matches how train.ex:728 and
# gen_micro_data.ex:579 locate repo-relative data.
csv = Path.join(File.cwd!(), ".cursor/research/cognitive_distortions_dataset.csv")
unless File.exists?(csv) do
  raise """
  cognitive-distortions dataset not found at #{csv}

  Run this with `mix run` from the umbrella root. There is no fallback corpus.
  """
end

neg_any = ~r/\b(not|never|no|nothing|nobody|neither|nor|none|without|cannot|hardly|barely|nowhere)\b|n't\b/i

texts =
  csv
  |> File.read!()
  |> String.split("\n", trim: true)
  |> Enum.drop(1)
  |> Enum.map(fn line ->
    case Regex.run(~r/^"((?:[^"]|"")*)"|^([^,]*)/, line) do
      [_, q] -> String.replace(q, "\"\"", "\"")
      [_, _, u] -> u
      _ -> ""
    end
  end)
  |> Enum.reject(&(String.trim(&1) == ""))

negated = Enum.filter(texts, &Regex.match?(neg_any, &1))
detected = Enum.filter(negated, &LinguisticData.has_negation?/1)
missed = negated -- detected

IO.puts("\n=== recall over the cognitive-distortions corpus ===")
IO.puts("  parsed              : #{length(texts)}")
IO.puts("  contain a negator   : #{length(negated)}")
IO.puts("  has_negation? true  : #{length(detected)}")
IO.puts("  MISSED              : #{length(missed)}  (#{Float.round(100 * length(missed) / max(length(negated), 1), 1)}%)")

IO.puts("\n  examples missed:")
missed |> Enum.take(8) |> Enum.each(fn t -> IO.puts("    #{inspect(String.slice(t, 0, 74))}") end)
