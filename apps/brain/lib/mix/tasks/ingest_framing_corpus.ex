defmodule Mix.Tasks.IngestFramingCorpus do
  @moduledoc """
  Extract the GVFC corpus from `GVFC.zip` in the project root into
  `data/framing/GVFC_extension_multimodal.csv`.

  The GVFC zip must be downloaded manually from
  <https://github.com/ganggit/GVFC-raw-corpus> (Google Drive link)
  and placed in the project root before running this task.

  ## Usage

      mix ingest_framing_corpus
  """

  use Mix.Task

  @shortdoc "Extract GVFC.zip into data/framing/"

  @zip_name "GVFC.zip"
  @target_dir "data/framing"
  @expected_csv "GVFC_extension_multimodal.csv"

  def run(_args) do
    root = File.cwd!()
    zip_path = Path.join(root, @zip_name)

    unless File.exists?(zip_path) do
      Mix.raise("""
      #{@zip_name} not found in the project root.

      Download it from https://github.com/ganggit/GVFC-raw-corpus
      (the repo README links to a Google Drive download).
      Place the zip file at: #{zip_path}

      Then rerun: mix ingest_framing_corpus
      """)
    end

    target = Path.join(root, @target_dir)
    dest = Path.join(target, @expected_csv)

    if File.exists?(dest) do
      Mix.shell().info("#{@expected_csv} already present at #{dest} — skipping extraction.")
    else
      File.mkdir_p!(target)
      Mix.shell().info("Extracting #{@zip_name}...")

      case :zip.unzip(String.to_charlist(zip_path), [{:cwd, String.to_charlist(target)}]) do
        {:ok, files} ->
          extracted =
            files
            |> Enum.map(&to_string/1)
            |> Enum.find(&String.ends_with?(&1, @expected_csv))

          if extracted && File.exists?(extracted) do
            if extracted != dest do
              File.rename!(extracted, dest)
            end

            Mix.shell().info("Extracted #{@expected_csv} to #{dest}")
            cleanup_macosx_dir(target)
            cleanup_nested_dir(target)
          else
            Mix.raise("""
            Extraction succeeded but #{@expected_csv} was not found in the archive.
            Files extracted: #{inspect(Enum.map(files, &to_string/1))}
            """)
          end

        {:error, reason} ->
          Mix.raise("Failed to extract #{@zip_name}: #{inspect(reason)}")
      end
    end
  end

  defp cleanup_macosx_dir(target) do
    macosx = Path.join(target, "__MACOSX")

    if File.dir?(macosx) do
      File.rm_rf!(macosx)
    end
  end

  defp cleanup_nested_dir(target) do
    nested = Path.join(target, "GVFC")

    if File.dir?(nested) do
      nested
      |> File.ls!()
      |> Enum.each(fn file ->
        File.rm(Path.join(nested, file))
      end)

      File.rmdir(nested)
    end
  end
end
