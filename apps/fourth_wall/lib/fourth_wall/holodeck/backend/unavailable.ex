defmodule FourthWall.Holodeck.Backend.Unavailable do
  @moduledoc """
  The default holodeck backend: no workspace exists, and every operation returns
  `{:error, :holodeck_unavailable}`.

  This is deliberate, not a stub. The write tier is a real capability with a real
  blast radius, so it is **off unless an operator turns it on** — a node with no
  Docker, a test run, or a deployment that has not opted in all get honest
  refusals rather than a fake workspace that silently drops writes. It mirrors
  the rest of the codebase's honesty rule (`:actuation_disabled`,
  `:corpus_not_indexed`): a capability that is not wired reports that it is not
  wired, so an officer never mistakes an unavailable tool for an empty result.
  """

  @behaviour FourthWall.Holodeck.Backend

  @unavailable {:error, :holodeck_unavailable}

  @impl true
  def create(_soul_id, _opts), do: @unavailable

  @impl true
  def destroy(_handle), do: :ok

  @impl true
  def read(_handle, _path), do: @unavailable

  @impl true
  def write(_handle, _path, _content), do: @unavailable

  @impl true
  def list(_handle, _path), do: @unavailable

  @impl true
  def exec(_handle, _argv), do: @unavailable
end
