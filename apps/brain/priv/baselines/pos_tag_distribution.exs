# Reports what the POS tagger actually emits, over the negation-bearing
# sentences of the cognitive-distortions set, alongside the tag vocabulary the
# trained model is capable of emitting.
#
#     mix run .claude/corpus/pos_tag_distribution.exs
#
# The gap between those two lists is task 074: measured 2026-09-12, the model's
# vocabulary holds 13 tags including VERB and PART, while the decoder emitted
# only 5 across 3,952 tokens — 91.4% NOUN, zero VERB, zero PART.

alias Brain.Analysis.Pipeline
alias Brain.ML.POSTagger

# `mix run` executes from the umbrella root; this matches how train.ex:728 and
# gen_micro_data.ex:579 locate repo-relative data.
csv = Path.join(File.cwd!(), ".cursor/research/cognitive_distortions_dataset.csv")

unless File.exists?(csv) do
  raise """
  cognitive-distortions dataset not found at #{csv}

  This script reports the tagger's emitted distribution over negation-bearing
  input. There is no fallback corpus.
  """
end
neg_re = ~r/\b(not|n't|never|no|nothing|nobody|neither|nor|cannot|can't|won't|don't|didn't|isn't|wasn't)\b/i

texts =
  csv |> File.read!() |> String.split("\n", trim: true) |> Enum.drop(1)
  |> Enum.map(fn line ->
    case Regex.run(~r/^"([^"]*)"|^([^,]*)/, line) do
      [_, q] -> q
      [_, _, u] -> u
      _ -> ""
    end
  end)
  |> Enum.filter(&(String.trim(&1) != "" and Regex.match?(neg_re, &1)))

IO.puts("negated sentences: #{length(texts)}")

all_tags =
  Enum.flat_map(texts, fn t ->
    a = Pipeline.analyze_chunk(t)

    (Map.get(a, :pos_tags) || [])
    |> Enum.map(fn
      {_w, tag} -> tag
      %{tag: tag} -> tag
      other -> other
    end)
  end)

IO.puts("\n=== tag distribution over #{length(all_tags)} tokens from negated sentences ===")

all_tags
|> Enum.frequencies()
|> Enum.sort_by(&(-elem(&1, 1)))
|> Enum.each(fn {tag, c} ->
  IO.puts("  #{String.pad_trailing(inspect(tag), 12)} #{String.pad_leading(to_string(c), 6)}  #{Float.round(100 * c / length(all_tags), 1)}%")
end)

IO.puts("\n=== the model's own tag_vocabulary (what it CAN emit) ===")

case POSTagger.load_model() do
  {:ok, m} -> IO.inspect(Map.keys(m.tag_vocabulary), label: "  tags in model")
  m when is_map(m) -> IO.inspect(Map.keys(m.tag_vocabulary), label: "  tags in model")
  other -> IO.inspect(other, label: "  load_model returned")
end

IO.puts("\n=== negation tokens specifically: what tag do they get? ===")
neg_words = ~w(not never no nothing nobody don't can't won't didn't isn't wasn't n't cannot)

Enum.take(texts, 8)
|> Enum.each(fn t ->
  a = Pipeline.analyze_chunk(t)
  tagged = Map.get(a, :pos_tags) || []

  hits =
    Enum.filter(tagged, fn
      {w, _} -> String.downcase(w) in neg_words
      %{token: w} -> String.downcase(w) in neg_words
      _ -> false
    end)

  if hits != [], do: IO.puts("  #{inspect(String.slice(t, 0, 50))} -> #{inspect(hits)}")
end)
