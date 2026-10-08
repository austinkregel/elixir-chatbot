#!/usr/bin/env elixir
#
# diagnostics/srl_spot_check.exs
#
# Runs ~20 representative utterances through Pipeline.analyze_chunk/1
# and inspects whether :srl_frames is populated. Helps determine if
# the SRL pipeline is producing data or consistently returning [].
#
# Run with:  mix run diagnostics/srl_spot_check.exs

alias Brain.Analysis.Pipeline

utterances = [
  "Turn on the living room lights",
  "Play some jazz music",
  "What time is it in Tokyo",
  "Set an alarm for 7 AM tomorrow",
  "Send a message to John",
  "How is the weather today",
  "Tell me a joke",
  "Create a reminder to buy groceries",
  "Open the garage door",
  "Who wrote Hamlet",
  "Add eggs to my shopping list",
  "Navigate to the nearest gas station",
  "What is the capital of France",
  "Book a table for two at seven",
  "I like chocolate ice cream",
  "Can you help me with my homework",
  "Dim the bedroom lights to 50 percent",
  "Read me the latest news",
  "Call Mom on her cell phone",
  "My name is Austin",
]

IO.puts("\n╔══════════════════════════════════════════════════════════════╗")
IO.puts("║            SRL SPOT CHECK — Pipeline.analyze_chunk          ║")
IO.puts("╚══════════════════════════════════════════════════════════════╝\n")

results =
  Enum.map(utterances, fn text ->
    try do
      analysis = Pipeline.analyze_chunk(text)
      srl_frames = Map.get(analysis, :srl_frames, [])
      pos_tags = Map.get(analysis, :pos_tags, [])
      {text, srl_frames, pos_tags}
    rescue
      e -> {text, {:error, Exception.message(e)}, []}
    end
  end)

populated = Enum.count(results, fn {_, frames, _} -> is_list(frames) and frames != [] end)
empty = Enum.count(results, fn {_, frames, _} -> frames == [] end)
errored = Enum.count(results, fn {_, frames, _} -> match?({:error, _}, frames) end)

IO.puts("Summary: #{populated} with frames, #{empty} empty, #{errored} errors\n")

Enum.each(results, fn {text, frames, pos_tags} ->
  IO.puts("─── \"#{text}\" ───")

  case frames do
    {:error, msg} ->
      IO.puts("  ERROR: #{msg}")

    [] ->
      IO.puts("  SRL frames: (none)")
      if pos_tags != [] do
        tags_str = pos_tags |> Enum.map(fn {tok, tag} -> "#{tok}/#{tag}" end) |> Enum.join(" ")
        IO.puts("  POS: #{tags_str}")
      end

    frames when is_list(frames) ->
      IO.puts("  SRL frames (#{length(frames)}):")
      Enum.each(frames, fn frame ->
        predicate = Map.get(frame, :predicate, Map.get(frame, "predicate", "?"))
        arguments = Map.get(frame, :arguments, Map.get(frame, "arguments", []))
        IO.puts("    predicate: #{inspect(predicate)}")
        Enum.each(arguments, fn arg ->
          role = Map.get(arg, :role, Map.get(arg, "role", "?"))
          text_span = Map.get(arg, :text, Map.get(arg, "text", "?"))
          IO.puts("      #{role}: \"#{text_span}\"")
        end)
      end)
  end

  IO.puts("")
end)
