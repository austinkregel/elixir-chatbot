defmodule Mix.Tasks.DownloadWordnet do
  @moduledoc """
  Downloads WordNet 3.1 Prolog database files for the Lexicon.

  ## Usage

      mix download_wordnet              # Download to priv/wordnet/
      mix download_wordnet --force      # Re-download even if files exist
      mix download_wordnet --dir path   # Custom output directory
  """

  use Mix.Task

  @shortdoc "Downloads WordNet 3.1 Prolog database for the Lexicon"

  @github_archive_url "https://github.com/ekaf/wordnet-prolog/archive/refs/heads/master.tar.gz"

  @required_files ~w(wn_s.pl wn_g.pl wn_hyp.pl wn_ant.pl wn_exc.pl wn_mm.pl wn_ms.pl wn_mp.pl wn_sim.pl wn_der.pl wn_ins.pl)

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [force: :boolean, dir: :string]
      )

    Application.ensure_all_started(:req)

    output_dir = Keyword.get(opts, :dir, default_output_dir())
    force = Keyword.get(opts, :force, false)

    if not force and files_present?(output_dir) do
      Mix.shell().info("WordNet files already present in #{output_dir}. Use --force to re-download.")
      :ok
    else
      download_and_extract(output_dir)
    end
  end

  defp default_output_dir do
    brain_priv = :code.priv_dir(:brain)

    if is_list(brain_priv) do
      Path.join(to_string(brain_priv), "wordnet")
    else
      Path.join("priv", "wordnet")
    end
  end

  defp files_present?(dir) do
    Enum.all?(@required_files, fn f -> File.exists?(Path.join(dir, f)) end)
  end

  defp download_and_extract(output_dir) do
    File.mkdir_p!(output_dir)

    tmp_path = Path.join(System.tmp_dir!(), "wordnet-prolog-#{System.system_time(:second)}.tar.gz")

    Mix.shell().info("Downloading WordNet 3.1 Prolog database from GitHub...")

    case Req.get(@github_archive_url, into: File.stream!(tmp_path), receive_timeout: 300_000) do
      {:ok, %{status: 200}} ->
        size = File.stat!(tmp_path).size
        Mix.shell().info("Downloaded #{format_size(size)}, extracting...")
        extract_prolog_files(tmp_path, output_dir)
        File.rm(tmp_path)

        if files_present?(output_dir) do
          Mix.shell().info("WordNet 3.1 Prolog files installed to #{output_dir}")
          count = output_dir |> File.ls!() |> Enum.count(&String.ends_with?(&1, ".pl"))
          Mix.shell().info("#{count} Prolog files extracted")
        else
          missing = Enum.reject(@required_files, &File.exists?(Path.join(output_dir, &1)))
          Mix.raise("Extraction incomplete. Missing files: #{Enum.join(missing, ", ")}")
        end

      {:ok, %{status: status}} ->
        File.rm(tmp_path)
        Mix.raise("Download failed: HTTP #{status}")

      {:error, reason} ->
        File.rm(tmp_path)
        Mix.raise("Download failed: #{inspect(reason)}")
    end
  end

  defp extract_prolog_files(tar_path, output_dir) do
    case :erl_tar.extract(String.to_charlist(tar_path), [:compressed, :memory]) do
      {:ok, file_list} ->
        file_list
        |> Enum.filter(fn {name, _content} ->
          name_str = to_string(name)
          String.ends_with?(name_str, ".pl") and not String.contains?(name_str, "/wn_query") and
            not String.contains?(name_str, "/wn_valid") and
            not String.contains?(name_str, "/wn_morphy") and
            not String.contains?(name_str, "/wn2csv") and
            not String.contains?(name_str, "/wn_syntax") and
            not String.contains?(name_str, "/utils") and
            not String.contains?(name_str, "/loader") and
            not String.contains?(name_str, "/timeit")
        end)
        |> Enum.each(fn {name, content} ->
          basename = Path.basename(to_string(name))
          dest = Path.join(output_dir, basename)
          File.write!(dest, content)
          Mix.shell().info("  #{basename} (#{format_size(byte_size(content))})")
        end)

      {:error, reason} ->
        Mix.raise("Failed to extract tar.gz: #{inspect(reason)}")
    end
  end

  defp format_size(bytes) when bytes < 1_024, do: "#{bytes} B"
  defp format_size(bytes) when bytes < 1_048_576, do: "#{Float.round(bytes / 1_024, 1)} KB"
  defp format_size(bytes), do: "#{Float.round(bytes / 1_048_576, 1)} MB"
end
