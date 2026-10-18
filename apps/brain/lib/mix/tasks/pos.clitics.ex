defmodule Mix.Tasks.Pos.Clitics do
  @shortdoc "Derive the English clitic table from the UD English Web Treebank"
  @moduledoc """
  Writes `priv/knowledge/clitics.json`, the clitics `Brain.Lexicon.Clitics`
  serves to the tokenizer and to `Brain.ML.InformalExpansions`.

      mix pos.clitics

  A clitic is a word the treebank writes fused onto the word before it (a
  non-first member of a multiword token: "n't" in "don't", "'s" in
  "Sarah's") whose form carries an apostrophe and a letter. The apostrophe
  is the written signal a tokenizer can split on; fused words without one
  ("s" in "its" meaning "it is", "na" in "gonna") cannot be told apart from
  whole words by spelling, so they are not clitics here.

  For each clitic the table records how often the treebank uses it and with
  which lemmas and parts of speech. Apostrophe variants the treebank writes
  (#{Enum.map_join(["’", "`", "´"], " ", &inspect/1)}) are counted under "'".

  Reads the EWT train split only, fetched and verified against its pin by
  `Brain.Training.UDEWT`; dev and test stay held out.
  """

  use Mix.Task

  alias Brain.Training.UDEWT

  @producer_version "1"
  @split "train"
  @apostrophes ["'", "’", "`", "´"]
  @out "knowledge/clitics.json"

  @impl Mix.Task
  def run(args) do
    {_opts, _, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [], do: Mix.raise("pos.clitics: unknown options #{inspect(invalid)}")

    source = UDEWT.source()
    %{"file" => file, "sha256" => sha} = Map.fetch!(source["files"], @split)

    clitics =
      @split
      |> UDEWT.fetch_verified!()
      |> UDEWT.parse!()
      |> Enum.flat_map(& &1.words)
      |> Enum.filter(&clitic?/1)
      |> Enum.group_by(&canonical(&1.form))
      |> Map.new(fn {form, words} ->
        {form,
         %{
           "count" => length(words),
           "lemmas" => Enum.frequencies_by(words, &canonical(&1.lemma)),
           "upos" => Enum.frequencies_by(words, & &1.upos)
         }}
      end)

    if clitics == %{}, do: Mix.raise("pos.clitics: the #{@split} split has no clitics; the parse is wrong")

    table = %{
      "format_version" => 1,
      "description" =>
        "English clitics: words the UD English Web Treebank fuses onto the word before them, " <>
          "with an apostrophe, and the lemmas it gives them. Written by `mix pos.clitics`.",
      "license" => source["license"],
      "produced_at" => DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601(),
      "producer" => %{
        "name" => inspect(__MODULE__),
        "version" => @producer_version,
        "inputs" => [%{"name" => "ud_ewt/#{file}", "version" => source["version"], "sha256" => sha}]
      },
      "apostrophes" => @apostrophes,
      "clitics" => clitics
    }

    out = Brain.priv_path(@out)
    File.write!(out, Jason.encode!(table, pretty: true) <> "\n")

    Mix.shell().info("#{map_size(clitics)} clitics from ud_ewt #{@split} -> #{out}")

    for {form, %{"count" => count, "lemmas" => lemmas}} <- Enum.sort_by(clitics, fn {_, c} -> -c["count"] end) do
      Mix.shell().info("  #{String.pad_trailing(form, 5)} #{String.pad_leading(to_string(count), 5)}  #{inspect(lemmas)}")
    end
  end

  defp clitic?(%{fused?: true, form: form}) do
    String.contains?(form, @apostrophes) and String.match?(form, ~r/\p{L}/u)
  end

  defp clitic?(_word), do: false

  defp canonical(text), do: text |> String.replace(@apostrophes, "'") |> String.downcase()
end
