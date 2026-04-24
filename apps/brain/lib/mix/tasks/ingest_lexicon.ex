defmodule Mix.Tasks.IngestLexicon do
  @moduledoc """
  Ingests external lexicon data (WordNet sense keys + ConceptNet) into
  pre-built `.term` files for fast startup loading.

  ## Usage

      mix ingest_lexicon                    # Build all lexicon data
      mix ingest_lexicon --skip-conceptnet  # Only process WordNet sense keys
      mix ingest_lexicon --conceptnet-path path/to/assertions.csv  # Use local ConceptNet file
      mix ingest_lexicon --max-entries 100000  # Limit ConceptNet entries

  ## Prerequisites

  - WordNet must be downloaded: `mix download_wordnet`
  - ConceptNet assertions CSV can be downloaded from:
    https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
  """

  use Mix.Task

  @shortdoc "Ingests WordNet + ConceptNet into pre-built lexicon files"

  @conceptnet_url "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          skip_conceptnet: :boolean,
          conceptnet_path: :string,
          max_entries: :integer,
          force: :boolean
        ]
      )

    Application.ensure_all_started(:jason)

    output_dir = output_dir()
    File.mkdir_p!(output_dir)

    Mix.shell().info("Lexicon ingestion output directory: #{output_dir}")

    unless Keyword.get(opts, :skip_conceptnet, false) do
      ingest_conceptnet(opts)
    end

    Mix.shell().info("Lexicon ingestion complete.")
  end

  defp ingest_conceptnet(opts) do
    csv_path = Keyword.get(opts, :conceptnet_path) || download_conceptnet(opts)
    max_entries = Keyword.get(opts, :max_entries, :infinity)
    output_path = Path.join(output_dir(), "conceptnet.term")

    if not Keyword.get(opts, :force, false) and File.exists?(output_path) do
      Mix.shell().info("ConceptNet data already exists at #{output_path}. Use --force to rebuild.")
    else
      Mix.shell().info("Parsing ConceptNet from #{csv_path}...")

      {:ok, count} =
        Brain.Lexicon.ConceptNetParser.parse_and_save(csv_path, output_path,
          max_entries: max_entries,
          min_weight: 1.0
        )

      Mix.shell().info("ConceptNet: #{count} concepts saved to #{output_path}")
    end
  end

  defp download_conceptnet(opts) do
    csv_path = Path.join(output_dir(), "conceptnet-assertions.csv")
    gz_path = csv_path <> ".gz"

    if File.exists?(csv_path) and not Keyword.get(opts, :force, false) do
      Mix.shell().info("ConceptNet CSV already exists at #{csv_path}")
      csv_path
    else
      Application.ensure_all_started(:req)

      Mix.shell().info("Downloading ConceptNet 5.7.0 assertions (~300MB compressed)...")
      Mix.shell().info("From: #{@conceptnet_url}")

      case Req.get(@conceptnet_url, into: File.stream!(gz_path), receive_timeout: 600_000) do
        {:ok, %{status: 200}} ->
          Mix.shell().info("Downloaded, decompressing...")
          decompress_gz(gz_path, csv_path)
          File.rm(gz_path)
          Mix.shell().info("ConceptNet CSV ready at #{csv_path}")
          csv_path

        {:ok, %{status: status}} ->
          File.rm(gz_path)
          Mix.raise("ConceptNet download failed: HTTP #{status}")

        {:error, reason} ->
          File.rm(gz_path)
          Mix.raise("ConceptNet download failed: #{inspect(reason)}")
      end
    end
  end

  defp decompress_gz(gz_path, output_path) do
    gz_data = File.read!(gz_path)
    decompressed = :zlib.gunzip(gz_data)
    File.write!(output_path, decompressed)
  end

  defp output_dir do
    priv_dir =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv"
        dir -> to_string(dir)
      end

    Path.join(priv_dir, "lexicon")
  end
end
