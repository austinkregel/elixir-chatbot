defmodule Brain.HTTP.Retry do
  @moduledoc """
  Shared Req retry instrumentation.

  Req's built-in retry handler emits a generic warning that doesn't tell
  you which service or URL is being polled:

      [warning] retry: got exception, will retry in 1000ms, 3 attempts left
      [warning] ** (Req.TransportError) connection refused

  When several HTTP clients are running on boot (Ouro sidecar health
  polling, model downloader, …) it's impossible to tell from the log
  which one is failing.

  This module replaces Req's default retry log with an enriched one and
  exposes a tiny helper, `options/2`, that returns the correct keyword
  options to splat into a `Req.get/post/...` call.

  ## Usage

      Req.get(url,
        Brain.HTTP.Retry.options("Ouro sidecar", url) ++
          [receive_timeout: 5_000]
      )

  Or, when you also want to override Req's defaults (e.g. cap retries):

      Req.get(url, Brain.HTTP.Retry.options("Ouro sidecar", url, max_retries: 1))

  ## Output

      [warning] HTTP retry [Ouro sidecar @ http://localhost:8100/health]: \\
        connection refused (attempt 1/4, retrying in 1000ms)

  ## What we log

  | Field      | Source                                                     |
  | ---------- | ---------------------------------------------------------- |
  | service    | First arg to `options/2` ("Ouro sidecar", "Atlas", ...)    |
  | url        | Second arg to `options/2`                                  |
  | reason     | `Req.TransportError.reason` or HTTP status                 |
  | attempt    | Derived from `request.private.req_retry_count`             |
  | max        | `:max_retries + 1` (Req counts retries, not total attempts)|
  | delay      | Computed from Req's exponential backoff                    |
  """

  require Logger

  @default_max_retries 3

  @doc """
  Returns Req options that swap in our enriched retry log.

  - `service` is a human-readable label like `"Ouro sidecar"`.
  - `url` is the request URL (used solely for the log; Req sets the URL
    independently from these options).
  - `opts` accepts:
    * `:max_retries` (default `3`)
    * `:retry_delay` (passes through to Req as a function or integer)
    * `:retry_log_level` (default `:warning`; pass `false` to silence)
  """
  @spec options(String.t(), String.t(), keyword()) :: keyword()
  def options(service, url, opts \\ []) do
    max_retries = Keyword.get(opts, :max_retries, @default_max_retries)
    retry_delay = Keyword.get(opts, :retry_delay, &default_delay/1)
    log_level = Keyword.get(opts, :retry_log_level, :warning)

    [
      retry: build_retry_fun(service, url, max_retries, retry_delay, log_level),
      retry_delay: retry_delay,
      max_retries: max_retries,
      retry_log_level: false
    ]
  end

  ## Internals ──────────────────────────────────────────────────────

  defp build_retry_fun(service, url, max_retries, retry_delay, log_level) do
    fn request, response_or_exception ->
      decision = default_retry_decision(request, response_or_exception)

      case decision do
        false ->
          false

        nil ->
          false

        true ->
          attempt = current_attempt(request)
          delay = compute_delay(retry_delay, attempt - 1)

          log_retry(
            log_level,
            service,
            url,
            response_or_exception,
            attempt,
            max_retries + 1,
            delay
          )

          true

        {:delay, ms} = result ->
          attempt = current_attempt(request)

          log_retry(
            log_level,
            service,
            url,
            response_or_exception,
            attempt,
            max_retries + 1,
            ms
          )

          result
      end
    end
  end

  # Mirrors Req's `:safe_transient` default — retry GETs/HEADs on:
  #   - `Req.TransportError` with reason in `[:timeout, :econnrefused, :closed]`
  #   - `Req.HTTPError` with `protocol: :http2, reason: :unprocessed`
  #   - HTTP 408/429/500/502/503/504
  defp default_retry_decision(request, response_or_exception) do
    method_safe? = request.method in [:get, :head]

    case response_or_exception do
      %Req.Response{status: status} when status in [408, 429, 500, 502, 503, 504] ->
        method_safe?

      %Req.TransportError{reason: reason} when reason in [:timeout, :econnrefused, :closed] ->
        method_safe?

      %Req.HTTPError{protocol: :http2, reason: :unprocessed} ->
        method_safe?

      _ ->
        false
    end
  end

  # Req stores the number of retries already performed in
  # `request.private.req_retry_count`. The first call (no retries yet)
  # has the key absent; the second call (after one retry) has `1`, etc.
  # We report the human-friendly attempt number (1-indexed) of the
  # *next* attempt that retrying will produce.
  defp current_attempt(request) do
    already = Map.get(request.private || %{}, :req_retry_count, 0)
    already + 2
  end

  defp compute_delay(fun, count) when is_function(fun, 1), do: fun.(count)
  defp compute_delay(ms, _count) when is_integer(ms), do: ms
  defp compute_delay(_, count), do: default_delay(count)

  defp default_delay(count), do: Integer.pow(2, count) * 1_000

  defp log_retry(false, _, _, _, _, _, _), do: :ok

  defp log_retry(level, service, url, reason, attempt, total, delay) do
    msg =
      "HTTP retry [#{service} @ #{url}]: #{format_reason(reason)} " <>
        "(attempt #{attempt}/#{total}, retrying in #{delay}ms)"

    Logger.log(level, msg)
  end

  defp format_reason(%Req.TransportError{reason: :econnrefused}), do: "connection refused"
  defp format_reason(%Req.TransportError{reason: :timeout}), do: "timeout"
  defp format_reason(%Req.TransportError{reason: :closed}), do: "connection closed"
  defp format_reason(%Req.TransportError{reason: reason}), do: "transport error: #{inspect(reason)}"

  defp format_reason(%Req.HTTPError{protocol: protocol, reason: reason}),
    do: "HTTP error (#{protocol}): #{inspect(reason)}"

  defp format_reason(%Req.Response{status: status}), do: "HTTP #{status}"
  defp format_reason(other), do: inspect(other)
end
