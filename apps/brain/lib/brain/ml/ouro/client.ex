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
    - `:max_tokens`        – max tokens to generate (default 256)
    - `:temperature`       – sampling temperature (default 1.0, per paper Table 17)
    - `:top_p`             – nucleus sampling (default 0.7, per paper Table 17)
    - `:repetition_penalty` – repetition penalty (default 1.2)
  """
  @spec chat_completion(list(map()), keyword()) :: {:ok, String.t()} | {:error, term()}
  def chat_completion(messages, opts \\ []) when is_list(messages) do
    url = api_url() <> "/v1/chat/completions"

    body = %{
      model: model_id(),
      messages: Enum.map(messages, &Map.take(&1, [:role, :content])),
      max_tokens: Keyword.get(opts, :max_tokens, 256),
      temperature: Keyword.get(opts, :temperature, 1.0),
      top_p: Keyword.get(opts, :top_p, 0.7),
      repetition_penalty: Keyword.get(opts, :repetition_penalty, 1.2)
    }

    start = System.monotonic_time(:millisecond)

    req_opts =
      [json: body, receive_timeout: request_timeout()] ++
        Brain.HTTP.Retry.options("Ouro sidecar", url)

    case Req.post(url, req_opts) do
      {:ok, %Req.Response{status: 200, body: resp}} ->
        elapsed = System.monotonic_time(:millisecond) - start
        raw = get_in(resp, ["choices", Access.at(0), "message", "content"]) || ""
        text = normalize_sidecar_message_content(raw)
        usage = resp["usage"] || %{}

        Logger.info(
          "Ouro sidecar: #{usage["completion_tokens"] || "?"} tokens in #{elapsed}ms " <>
            "(prompt=#{usage["prompt_tokens"] || "?"}, total=#{usage["total_tokens"] || "?"})"
        )

        {:ok, text}

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

    # Health checks are inherently driven by an outer periodic poller
    # (SidecarLauncher's `:health_poll` loop), so Req-level retries are
    # redundant *and* spammy: each retry produces another `connection
    # refused` log line during the 5–10s window Python takes to load.
    # Use `max_retries: 0` so a single call is a single network attempt
    # with at most one failure log; the outer poller decides cadence.
    req_opts =
      [receive_timeout: 5_000] ++
        Brain.HTTP.Retry.options("Ouro sidecar (health)", url, max_retries: 0)

    case Req.get(url, req_opts) do
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

    req_opts =
      [receive_timeout: 5_000] ++
        Brain.HTTP.Retry.options("Ouro sidecar (models)", url, max_retries: 1)

    case Req.get(url, req_opts) do
      {:ok, %Req.Response{status: 200, body: %{"data" => models}}} ->
        {:ok, models}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # --- Sidecar content normalization ---

  # Models occasionally emit JSON-encoded strings or a trailing `}`; unwrap so
  # callers receive plain assistant text.
  defp normalize_sidecar_message_content(raw) when is_binary(raw) do
    raw
    |> String.trim()
    |> unwrap_json_string_layers()
  end

  defp normalize_sidecar_message_content(_), do: ""

  defp unwrap_json_string_layers(text) do
    case Jason.decode(text) do
      {:ok, inner} when is_binary(inner) ->
        inner |> String.trim() |> unwrap_json_string_layers()

      _ ->
        trimmed = text |> String.trim_trailing("}") |> String.trim()

        if trimmed != text do
          unwrap_json_string_layers(trimmed)
        else
          text
        end
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
    gen_timeout = ml_config[:ouro_generation_timeout] || @default_timeout
    # HTTP must fail before the GenServer call timeout so the GenServer
    # receives a clean {:error, _} instead of a :timeout crash.
    max(gen_timeout - 5_000, 5_000)
  end
end
