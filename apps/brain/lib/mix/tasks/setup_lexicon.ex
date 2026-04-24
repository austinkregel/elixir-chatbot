defmodule Mix.Tasks.SetupLexicon do
  @moduledoc """
  Downloads and ingests all lexicon data (WordNet + ConceptNet) in one command.

  This is the recommended way to set up the lexicon for a fresh install or
  after clearing build artifacts. It runs both `mix download_wordnet` and
  `mix ingest_lexicon` sequentially.

  ## Usage

      mix setup_lexicon                    # Full setup (WordNet + ConceptNet)
      mix setup_lexicon --skip-conceptnet  # Only set up WordNet
      mix setup_lexicon --force            # Re-download and rebuild everything

  ## What it does

  1. Downloads WordNet 3.1 Prolog database (`mix download_wordnet`)
  2. Downloads and ingests ConceptNet 5.7.0 (`mix ingest_lexicon`)

  Both data sources are required for full feature vector coverage. Without
  ConceptNet, the 12 ConceptNet edge-type dimensions (218-229) will be
  zero. Without WordNet, lexical domains, supersenses, and selectional
  preferences will be unavailable.
  """

  use Mix.Task

  @shortdoc "Download and ingest all lexicon data (WordNet + ConceptNet)"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [skip_conceptnet: :boolean, force: :boolean]
      )

    force_args = if Keyword.get(opts, :force, false), do: ["--force"], else: []

    Mix.shell().info("""

    ╔══════════════════════════════════════════════════════════════╗
    ║                   Lexicon Setup                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Step 1: Download WordNet 3.1 Prolog database              ║
    ║  Step 2: Ingest ConceptNet 5.7.0 assertions                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    Mix.shell().info("── Step 1: WordNet ──────────────────────────────────────────")
    Mix.Task.run("download_wordnet", force_args)

    unless Keyword.get(opts, :skip_conceptnet, false) do
      Mix.shell().info("\n── Step 2: ConceptNet ───────────────────────────────────────")
      Mix.Task.run("ingest_lexicon", force_args)
    end

    Mix.shell().info("\nLexicon setup complete.")
    check_status()
  end

  defp check_status do
    priv_dir =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv"
        dir -> to_string(dir)
      end

    wordnet_dir = Path.join(priv_dir, "wordnet")
    conceptnet_path = Path.join(priv_dir, "lexicon/conceptnet.term")

    wordnet_ok = File.dir?(wordnet_dir) and File.exists?(Path.join(wordnet_dir, "wn_s.pl"))
    conceptnet_ok = File.exists?(conceptnet_path)

    Mix.shell().info("\n── Status ──────────────────────────────────────────────────")
    Mix.shell().info("  WordNet:    #{if wordnet_ok, do: "OK", else: "MISSING"}")
    Mix.shell().info("  ConceptNet: #{if conceptnet_ok, do: "OK", else: "MISSING"}")

    unless wordnet_ok and conceptnet_ok do
      Mix.shell().info("\n  Some lexicon data is missing. Feature vectors will have dead dimensions.")
    end
  end
end
