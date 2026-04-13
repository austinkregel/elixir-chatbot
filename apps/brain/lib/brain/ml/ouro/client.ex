defmodule Brain.ML.Ouro.Client do
  @moduledoc """
  HTTP client for the Ouro inference sidecar.

  Communicates with the Ouro server (scripts/ouro_server.py) via the
  OpenAI-compatible `/v1/chat/completions` API.  Returns the same
  `{:ok, text}` / `{:error, reason}` shape that the rest of the
  pipeline expects from `Ouro.Model.generate/2`.
  """

  require Logger

  @default_url "http://localhost:8100"
  @default_timeout 60_000

  @doc """
  Sends a chat completion request and returns the generated text.

  `messages` is a list of `%{role: role, content: content}` maps
  (same format that `RealizationPacket.build/3` produces).

  Options:
    - `:max_tokens`  – max tokens to generate (default 256)
    - `:temperature` – sampling temperature (default 0.7)
    - `:top_p`       – nucleus sampling (default 0.9)
  """
  @spec chat_completion(list(map()), keyword()) :: {:ok, String.t()} | {:error, term()}
  def chat_completion(messages, opts \\ []) when is_list(messages) do
    url = api_url() <> "/v1/chat/completions"

    body = %{
      model: model_id(),
      messages: Enum.map(messages, &Map.take(&1, [:role, :content])),
      max_tokens: Keyword.get(opts, :max_tokens, 256),
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.9)
    }

    start = System.monotonic_time(:millisecond)

    case Req.post(url, json: body, receive_timeout: request_timeout()) do
      {:ok, %Req.Response{status: 200, body: resp}} ->
        elapsed = System.monotonic_time(:millisecond) - start
        text = get_in(resp, ["choices", Access.at(0), "message", "content"]) || ""
        usage = resp["usage"] || %{}

        Logger.info(
          "Ouro sidecar: #{usage["completion_tokens"] || "?"} tokens in #{elapsed}ms " <>
            "(prompt=#{usage["prompt_tokens"] || "?"}, total=#{usage["total_tokens"] || "?"})"
        )

        {:ok, String.trim(text)}

      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ouro sidecar HTTP #{status}: #{inspect(body)}")
        {:error, {:http_error, status, body}}

      {:error, %Req.TransportError{reason: :econnrefused}} ->
        Logger.warning("Ouro sidecar not reachable at #{url}")
        {:error, :sidecar_unavailable}

      {:error, reason} ->
        Logger.error("Ouro sidecar request failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Checks whether the sidecar is reachable and serving the model.
  """
  @spec health_check() :: :ok | {:error, term()}
  def health_check do
    url = api_url() <> "/health"

    case Req.get(url, receive_timeout: 5_000) do
      {:ok, %Req.Response{status: 200, body: %{"status" => "ok"}}} ->
        :ok

      {:ok, %Req.Response{status: status}} ->
        {:error, {:unhealthy, status}}

      {:error, %Req.TransportError{reason: :econnrefused}} ->
        {:error, :sidecar_unavailable}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Lists available models from the sidecar.
  """
  @spec list_models() :: {:ok, list(map())} | {:error, term()}
  def list_models do
    url = api_url() <> "/v1/models"

    case Req.get(url, receive_timeout: 5_000) do
      {:ok, %Req.Response{status: 200, body: %{"data" => models}}} ->
        {:ok, models}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # --- Config helpers ---

  defp api_url do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_api_url] || @default_url
  end

  defp model_id do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_model_id] || "ByteDance/Ouro-2.6B"
  end

  defp request_timeout do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_generation_timeout] || @default_timeout
  end
end
