defmodule Mix.Tasks.Pos.ImportUdEwt do
  @shortdoc "Import the UD English Web Treebank as POS training fixtures"
  @moduledoc """
  Builds POS training fixtures (task 086 format) from the Universal
  Dependencies English Web Treebank, the POS training source.

      mix pos.import_ud_ewt

  1. Reads the pinned release from `priv/training/pos/sources.json` and
     fetches each file into `data/corpora/ud_ewt/<version>/` unless it is
     already there (`Brain.Training.UDEWT`).
  2. Verifies every file's SHA-256 against the manifest. A mismatch fails the
     import: different data under the same release name is never used.
  3. Converts each split to `priv/training/pos/ud_ewt.<split>.json`, mapping
     UD v2 tags onto the tagger's ud_v1 through the declared
     `priv/training/pos/ud_v2_to_ud_v1.json`. A tag with no mapping fails.
  4. Reloads each written file through `Brain.Training.Fixture.load!/1`, so
     an import that breaks the format fails here rather than at training.

  Tokens are the treebank's syntactic words: a fused token such as "don't"
  appears as "do" + "n't", each with its own tag. Empty nodes of the enhanced
  graph are not words and are skipped.
  """

  use Mix.Task

  alias Brain.Training.UDEWT

  @fixture_dir "training/pos"
  @producer_version "1"

  @impl Mix.Task
  def run(args) do
    {_opts, _, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [], do: Mix.raise("pos.import_ud_ewt: unknown options #{inspect(invalid)}")

    fixture_dir = Brain.priv_path(@fixture_dir)
    source = UDEWT.source()
    mapping_path = Path.join(fixture_dir, "ud_v2_to_ud_v1.json")
    mapping = mapping_path |> File.read!() |> Jason.decode!() |> Map.fetch!("map")
    mapping_sha = UDEWT.sha256_file(mapping_path)
    produced_at = DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()

    for {split, %{"file" => file, "sha256" => sha}} <- Enum.sort(source["files"]) do
      path = UDEWT.fetch_verified!(split)

      producer = %{
        "name" => inspect(__MODULE__),
        "version" => @producer_version,
        "inputs" => [
          %{"name" => "ud_ewt/#{file}", "version" => source["version"], "sha256" => sha},
          %{
            "name" => "ud_v2_to_ud_v1",
            "version" => "repo:apps/brain/priv/#{@fixture_dir}/ud_v2_to_ud_v1.json",
            "sha256" => mapping_sha
          }
        ]
      }

      records =
        path
        |> UDEWT.parse!()
        |> Enum.map(&to_record(&1, file, source["license"], produced_at, producer, mapping))

      out = Path.join(fixture_dir, "ud_ewt.#{split}.json")
      write_fixture!(out, records)
      loaded = Brain.Training.Fixture.load!(out)
      words = loaded |> Enum.map(&length(&1["tokens"])) |> Enum.sum()

      Mix.shell().info("#{split}: #{length(loaded)} sentences, #{words} words -> #{out}")
    end
  end

  defp to_record(sentence, file, license, produced_at, producer, mapping) do
    source_id = "#{file}##{sentence.sent_id}"
    digest = :crypto.hash(:sha256, source_id <> "\n" <> sentence.text) |> Base.encode16(case: :lower)

    tags =
      Enum.map(sentence.words, fn %{form: form, upos: upos} ->
        Map.get(mapping, upos) ||
          Mix.raise("pos.import_ud_ewt: #{source_id}: #{inspect(form)} has tag #{upos}, which ud_v2_to_ud_v1 does not map")
      end)

    %{
      "id" => "ud_ewt/" <> binary_part(digest, 0, 16),
      "text" => sentence.text,
      "origin" => "corpus:ud_ewt",
      "source_id" => source_id,
      "license" => license,
      "produced_at" => produced_at,
      "producer" => producer,
      "tokens" => Enum.map(sentence.words, & &1.form),
      "layers" => %{
        "pos" => %{
          "tags" => tags,
          "vocabulary" => "ud_v1",
          "resolution" => List.duplicate("corpus", length(tags))
        }
      }
    }
  end

  # One record per line, so a re-import diffs record by record.
  defp write_fixture!(path, records) do
    body = Enum.map_join(records, ",\n", &Jason.encode!/1)
    File.write!(path, ~s({"format_version": 1, "records": [\n) <> body <> "\n]}\n")
  end
end
