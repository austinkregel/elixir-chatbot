defmodule Brain.ML.Ouro.Tokenizer do
  @moduledoc """
  Extends Bumblebee's Llama tokenizer loading with Ouro-specific special tokens.

  Ouro uses ChatML-style tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`)
  which don't map to any single Bumblebee tokenizer type. We load via `:llama`
  to get the native Rust tokenizer initialized from `tokenizer.json`, then patch
  the special token map using the actual values from `tokenizer_config.json` so
  it works correctly for both Thinking and non-Thinking model variants.
  """

  require Logger

  @token_keys [
    {:bos, "bos_token"},
    {:eos, "eos_token"},
    {:unk, "unk_token"},
    {:pad, "pad_token"}
  ]

  @doc """
  Loads an Ouro tokenizer from a Bumblebee repository reference.

  Uses Llama as the base tokenizer type, then patches special tokens
  from the model's `tokenizer_config.json`.
  """
  def load(repo) do
    with {:ok, tokenizer} <- Bumblebee.load_tokenizer(repo, type: :llama) do
      special_tokens = read_special_tokens(repo)
      {:ok, %{tokenizer | special_tokens: Map.merge(tokenizer.special_tokens, special_tokens)}}
    end
  end

  defp read_special_tokens({:local, dir}) do
    config_path = Path.join(dir, "tokenizer_config.json")

    case File.read(config_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, data} -> extract_tokens(data)
          {:error, _} -> fallback_tokens()
        end

      {:error, _} ->
        fallback_tokens()
    end
  end

  defp read_special_tokens(_repo), do: fallback_tokens()

  defp extract_tokens(data) do
    for {atom_key, json_key} <- @token_keys,
        token = data[json_key],
        is_binary(token),
        into: %{} do
      {atom_key, token}
    end
    |> then(fn tokens ->
      if tokens[:pad] == nil and tokens[:eos] do
        Map.put(tokens, :pad, tokens[:eos])
      else
        tokens
      end
    end)
  end

  defp fallback_tokens do
    Logger.warning("Ouro.Tokenizer: could not read tokenizer_config.json, using defaults")

    %{
      bos: "<|im_start|>",
      eos: "<|im_end|>",
      unk: "<|endoftext|>",
      pad: "<|im_end|>"
    }
  end
end
