defmodule Mix.Tasks.Ouro.Download do
  @moduledoc """
  Downloads the Ouro LoopLM model files from HuggingFace.

  Weights are large (on the order of several GiB, e.g. ~5.3 GB for 2.6B). A truncated
  `model.safetensors` will pass a naive "file exists" check but makes Bumblebee/Safetensors
  fail with `:eof` when loading tensors — use `mix ouro.verify` after download.

  ## Usage

      mix ouro.download                # Download Ouro-1.4B (default, non-thinking)
      mix ouro.download --thinking     # Download Ouro-1.4B-Thinking
      mix ouro.download --model 2.6b   # Download Ouro-2.6B
      mix ouro.download --force        # Re-download even if files exist
      mix ouro.download --dir path     # Custom output directory
  """

  use Mix.Task

  alias Brain.ML.Ouro.ModelDownloader

  @shortdoc "Downloads Ouro LoopLM model from HuggingFace"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [model: :string, force: :boolean, dir: :string, thinking: :boolean]
      )

    model_size = Keyword.get(opts, :model, "1.4b")
    thinking = Keyword.get(opts, :thinking, false)

    unless model_size in ["1.4b", "2.6b"] do
      Mix.raise("Unknown model: #{model_size}. Use '1.4b' or '2.6b'.")
    end

    download_opts = [
      force: Keyword.get(opts, :force, false),
      thinking: thinking
    ]

    download_opts =
      case Keyword.get(opts, :dir) do
        nil -> download_opts
        dir -> Keyword.put(download_opts, :output_dir, dir)
      end

    repo = ModelDownloader.hf_repo(model_size, thinking)
    Mix.shell().info("Downloading #{repo} ...")

    case ModelDownloader.download(model_size, download_opts) do
      {:ok, dir} ->
        Mix.shell().info("Model files saved to #{dir}")

      {:error, errors} ->
        Mix.shell().error("Download failed:")

        Enum.each(errors, fn {:error, file, reason} ->
          Mix.shell().error("  #{file}: #{inspect(reason)}")
        end)
    end
  end
end
