defmodule Mix.Tasks.Pos.ImportUdEwt do
  @shortdoc "Import the UD English Web Treebank as POS training fixtures"
  @moduledoc """
  Builds POS training fixtures (task 086 format) from the Universal
  Dependencies English Web Treebank, the POS training source.

      mix pos.import_ud_ewt

  1. Reads the pinned release from `priv/training/pos/sources.json` and
     fetches each file into `data/corpora/ud_ewt/<version>/` unless it is
     already there.
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

  @fixture_dir "training/pos"
  @producer_version "1"

  @impl Mix.Task
  def run(args) do
    {_opts, _, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [], do: Mix.raise("pos.import_ud_ewt: unknown options #{inspect(invalid)}")

    Application.ensure_all_started(:req)

    fixture_dir = Brain.priv_path(@fixture_dir)
    source = fixture_dir |> Path.join("sources.json") |> read_json!() |> Map.fetch!("ud_ewt")
    mapping_path = Path.join(fixture_dir, "ud_v2_to_ud_v1.json")
    mapping = mapping_path |> read_json!() |> Map.fetch!("map")
    mapping_sha = sha256_file(mapping_path)
    cache_dir = Path.join([data_path(), "corpora", "ud_ewt", source["version"]])
    produced_at = DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()

    for {split, %{"file" => file, "sha256" => sha}} <- Enum.sort(source["files"]) do
      path = fetch!(source["raw_base"], file, cache_dir)
      verify!(path, sha)

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
        |> parse_conllu!()
        |> Enum.map(&to_record(&1, file, source["license"], produced_at, producer, mapping))

      out = Path.join(fixture_dir, "ud_ewt.#{split}.json")
      write_fixture!(out, records)
      loaded = Brain.Training.Fixture.load!(out)
      words = loaded |> Enum.map(&length(&1["tokens"])) |> Enum.sum()

      Mix.shell().info("#{split}: #{length(loaded)} sentences, #{words} words -> #{out}")
    end
  end

  defp data_path do
    Application.fetch_env!(:brain, :ml) |> Keyword.fetch!(:training_data_path)
  end

  defp read_json!(path), do: path |> File.read!() |> Jason.decode!()

  defp fetch!(raw_base, file, cache_dir) do
    path = Path.join(cache_dir, file)

    unless File.exists?(path) do
      File.mkdir_p!(cache_dir)
      url = "#{raw_base}/#{file}"
      Mix.shell().info("Fetching #{url}")

      case Req.get(url, into: File.stream!(path), receive_timeout: 300_000) do
        {:ok, %{status: 200}} ->
          :ok

        {:ok, %{status: status}} ->
          File.rm(path)
          Mix.raise("pos.import_ud_ewt: #{url} returned HTTP #{status}")

        {:error, reason} ->
          File.rm(path)
          Mix.raise("pos.import_ud_ewt: could not fetch #{url}: #{inspect(reason)}")
      end
    end

    path
  end

  defp verify!(path, expected) do
    actual = sha256_file(path)

    unless actual == expected do
      Mix.raise(
        "pos.import_ud_ewt: #{path} has sha256 #{actual}, but sources.json pins #{expected}. " <>
          "Delete the file to fetch it again, or update the pin if the release changed deliberately."
      )
    end
  end

  defp sha256_file(path) do
    path |> File.stream!(2_048) |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  # Returns [%{sent_id, text, words: [{form, upos}]}] in file order.
  defp parse_conllu!(path) do
    path
    |> File.read!()
    |> String.split(~r/\n\s*\n/, trim: true)
    |> Enum.map(&parse_sentence!(&1, path))
  end

  defp parse_sentence!(block, path) do
    lines = String.split(block, "\n", trim: true)
    meta = for "# " <> rest <- lines, [k, v] = String.split(rest, " = ", parts: 2), into: %{}, do: {k, v}

    words =
      for line <- lines, not String.starts_with?(line, "#"), word = parse_word!(line, path), word != :skip do
        word
      end

    sent_id = Map.get(meta, "sent_id") || Mix.raise("pos.import_ud_ewt: #{path}: a sentence has no sent_id")
    text = Map.get(meta, "text") || Mix.raise("pos.import_ud_ewt: #{path}: #{sent_id} has no text")
    if words == [], do: Mix.raise("pos.import_ud_ewt: #{path}: #{sent_id} has no words")

    %{sent_id: sent_id, text: text, words: words}
  end

  defp parse_word!(line, path) do
    case String.split(line, "\t") do
      [id, form, _lemma, upos | _rest] = cols when length(cols) == 10 ->
        cond do
          # A fused token ("don't" over "do" + "n't"): its words follow.
          String.contains?(id, "-") -> :skip
          # An empty node of the enhanced graph: not a word.
          String.contains?(id, ".") -> :skip
          upos == "_" -> Mix.raise("pos.import_ud_ewt: #{path}: word #{inspect(form)} has no UPOS")
          true -> {form, upos}
        end

      _ ->
        Mix.raise("pos.import_ud_ewt: #{path}: malformed line #{inspect(line)}")
    end
  end

  defp to_record(sentence, file, license, produced_at, producer, mapping) do
    source_id = "#{file}##{sentence.sent_id}"
    digest = :crypto.hash(:sha256, source_id <> "\n" <> sentence.text) |> Base.encode16(case: :lower)

    tags =
      Enum.map(sentence.words, fn {form, upos} ->
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
      "tokens" => Enum.map(sentence.words, &elem(&1, 0)),
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
