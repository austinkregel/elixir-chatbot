defmodule Brain.ML.Generation.OllamaAdmin do
  @moduledoc """
  Management client for a local Ollama server's *native* API
  (`/api/tags`, `/api/pull`) — distinct from `Brain.ML.Generation.OpenAICompatible`,
  which only speaks the OpenAI-compatible `/v1/chat/completions` surface Ollama
  also exposes. This module exists so the Fleet console can list/pull models and
  pick which one generation targets; it is not itself a `Generation.Backend`.

  Derives Ollama's root URL from the same `openai_compatible.base_url` config
  `OpenAICompatible` already uses (stripping the `/v1` suffix), so there's one
  place to point at a different Ollama host, not two.
  """
  require Logger

  @default_root "http://localhost:11434"
  # Model pulls are slow (multi-GB downloads) — this must outlast that, not the
  # ordinary generation-request timeout.
  @pull_timeout 1_800_000

  @doc "Local models this Ollama server already has pulled."
  def list_models do
    url = root() <> "/api/tags"

    case Req.get(url, receive_timeout: 5_000) do
      {:ok, %Req.Response{status: 200, body: %{"models" => models}}} ->
        {:ok, Enum.map(models, &to_model_info/1)}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, %Req.TransportError{reason: :econnrefused}} ->
        {:error, :ollama_unreachable}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Pulls a model by name. Blocking, and can take a long time for a large model
  — call this from a supervised Task, never inline in a LiveView callback.
  """
  def pull_model(name) when is_binary(name) and name != "" do
    url = root() <> "/api/pull"

    case Req.post(url, json: %{name: name, stream: false}, receive_timeout: @pull_timeout) do
      {:ok, %Req.Response{status: 200, body: %{"status" => "success"}}} ->
        :ok

      {:ok, %Req.Response{status: 200, body: body}} ->
        {:error, {:unexpected_response, body}}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, %Req.TransportError{reason: :econnrefused}} ->
        {:error, :ollama_unreachable}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "The model name `Brain.ML.Generation.OpenAICompatible` currently targets."
  def current_model do
    Brain.ML.Generation.config() |> Keyword.get(:openai_compatible, []) |> Keyword.get(:model)
  end

  @doc """
  Switches which model `openai_compatible` targets, at runtime — no restart,
  takes effect on the next generation call. Does not verify the model has
  actually been pulled; a bad name simply fails the next `generate/2` call
  loudly, same as any other backend-unavailable outcome.
  """
  def set_model(name) when is_binary(name) and name != "" do
    gen_cfg = Application.get_env(:brain, :generation, [])
    oc_cfg = Keyword.get(gen_cfg, :openai_compatible, [])

    Application.put_env(
      :brain, :generation,
      Keyword.put(gen_cfg, :openai_compatible, Keyword.put(oc_cfg, :model, name))
    )

    :ok
  end

  defp to_model_info(m) do
    %{name: m["name"], size: m["size"], modified_at: m["modified_at"]}
  end

  defp root do
    base =
      Brain.ML.Generation.config()
      |> Keyword.get(:openai_compatible, [])
      |> Keyword.get(:base_url, @default_root <> "/v1")

    base |> String.trim_trailing("/") |> String.trim_trailing("/v1")
  end
end
