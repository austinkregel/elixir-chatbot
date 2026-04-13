defmodule Brain.ML.Ouro.ModelDownloader do
  @moduledoc """
  Downloads Ouro model files from HuggingFace.

  Fetches the safetensors weights, tokenizer files, and config
  into `priv/ml_models/ouro/` for local inference.
  """

  require Logger

  @hf_base_url "https://huggingface.co"
  @default_output_dir "priv/ml_models/ouro"

  @hf_repos %{
    "1.4b" => "ByteDance/Ouro-1.4B",
    "1.4b-thinking" => "ByteDance/Ouro-1.4B-Thinking",
    "2.6b" => "ByteDance/Ouro-2.6B",
    "2.6b-thinking" => "ByteDance/Ouro-2.6B-Thinking"
  }

  @required_files [
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json"
  ]

  def hf_repo(model_size, thinking \\ false) do
    Map.fetch!(@hf_repos, model_size <> (if thinking, do: "-thinking", else: ""))
  end

  def required_files, do: @required_files

  # Full bf16 safetensors on HuggingFace are multi‑GiB (e.g. ~5.3 GB for 2.6B, ~2.5+ GB for 1.4B).
  # Truncated downloads or Git LFS pointer files are often a few hundred MB and make
  # Safetensors/Nx read past EOF → `:eof` in FileTensor.
  @min_safetensors_bytes 2_000_000_000

  @doc """
  Validates `model.safetensors` on disk.

  Returns `:ok` or `{:error, message}` when the file is missing, is a Git LFS pointer,
  or is far smaller than published Ouro weights (truncated / wrong artifact).
  """
  def validate_model_safetensors(path) when is_binary(path) do
    cond do
      not File.exists?(path) ->
        {:error, "file does not exist"}

      git_lfs_pointer_file?(path) ->
        {:error,
         "looks like a Git LFS pointer, not tensor data — use `mix ouro.download` " <>
           "(HF resolve URL) or `huggingface-cli download`, not raw `git clone` without LFS pull"}

      true ->
        size = File.stat!(path).size

        if size < @min_safetensors_bytes do
          {:error,
           "only #{format_size(size)} on disk; full Ouro weights from HuggingFace are " <>
             "several GiB (e.g. ~5.3 GB for 2.6B). File is truncated or not the real checkpoint — " <>
             "run `mix ouro.download --model 2.6b --force` (or `1.4b`) and wait for completion"}
        else
          :ok
        end
    end
  end

  @doc """
  Downloads all required model files for the given model size.

  Options:
    - `:output_dir` - destination directory (default: priv/ml_models/ouro)
    - `:force` - re-download even if files exist (default: false)
    - `:thinking` - use the Thinking variant (default: false)
  """
  def download(model_size \\ "1.4b", opts \\ []) do
    output_dir = Keyword.get(opts, :output_dir, models_dir())
    force = Keyword.get(opts, :force, false)
    thinking = Keyword.get(opts, :thinking, false)

    File.mkdir_p!(output_dir)
    repo = hf_repo(model_size, thinking)

    results =
      @required_files
      |> Enum.map(fn filename ->
        dest = Path.join(output_dir, filename)

        cond do
          filename == "model.safetensors" and File.exists?(dest) and not force ->
            case validate_model_safetensors(dest) do
              :ok ->
                Logger.info("#{filename} already exists, skipping")
                {:ok, filename}

              {:error, reason} ->
                Logger.error(
                  "#{filename} exists but is invalid (#{reason}). " <>
                    "Remove it or run with --force to re-download."
                )

                {:error, filename, reason}
            end

          File.exists?(dest) and not force ->
            Logger.info("#{filename} already exists, skipping")
            {:ok, filename}

          true ->
            download_file(repo, filename, dest)
        end
      end)

    errors = Enum.filter(results, &match?({:error, _, _}, &1))

    if errors == [] do
      Logger.info("All Ouro model files downloaded to #{output_dir}")
      {:ok, output_dir}
    else
      {:error, errors}
    end
  end

  @doc """
  Checks whether all required model files are present.
  """
  def files_present?(output_dir \\ nil) do
    dir = output_dir || models_dir()

    @required_files
    |> Enum.all?(fn filename ->
      File.exists?(Path.join(dir, filename))
    end)
  end

  defp download_file(repo, filename, dest) do
    url = "#{@hf_base_url}/#{repo}/resolve/main/#{filename}"
    Logger.info("Downloading #{filename} from #{url}")

    case Req.get(url, into: File.stream!(dest), receive_timeout: 600_000) do
      {:ok, %{status: 200}} ->
        size = File.stat!(dest).size
        Logger.info("Downloaded #{filename} (#{format_size(size)})")

        if filename == "model.safetensors" do
          case validate_model_safetensors(dest) do
            :ok ->
              {:ok, filename}

            {:error, reason} ->
              File.rm(dest)
              Logger.error("Downloaded #{filename} failed validation: #{reason}")
              {:error, filename, reason}
          end
        else
          {:ok, filename}
        end

      {:ok, %{status: status}} ->
        File.rm(dest)
        Logger.error("Failed to download #{filename}: HTTP #{status}")
        {:error, filename, "HTTP #{status}"}

      {:error, reason} ->
        File.rm(dest)
        Logger.error("Failed to download #{filename}: #{inspect(reason)}")
        {:error, filename, reason}
    end
  end

  defp git_lfs_pointer_file?(path) do
    prefix =
      case File.open(path, [:read, :binary], fn io -> IO.binread(io, 160) end) do
        {:error, _} -> ""
        bin when is_binary(bin) -> bin
        _ -> ""
      end

    String.starts_with?(prefix, "version https://git-lfs.") or
      (byte_size(prefix) < 500 and String.contains?(prefix, "git-lfs"))
  end

  defp models_dir do
    brain_priv = :code.priv_dir(:brain)

    if is_list(brain_priv) do
      Path.join(to_string(brain_priv), "ml_models/ouro")
    else
      Path.join([@default_output_dir])
    end
  end

  defp format_size(bytes) when bytes < 1_024, do: "#{bytes} B"
  defp format_size(bytes) when bytes < 1_048_576, do: "#{Float.round(bytes / 1_024, 1)} KB"
  defp format_size(bytes) when bytes < 1_073_741_824, do: "#{Float.round(bytes / 1_048_576, 1)} MB"
  defp format_size(bytes), do: "#{Float.round(bytes / 1_073_741_824, 2)} GB"
end
