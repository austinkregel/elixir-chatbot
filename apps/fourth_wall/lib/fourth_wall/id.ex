defmodule FourthWall.ID do
  @moduledoc """
  Shared ID generation utilities for the ChatBot umbrella.

  Uses cryptographically secure random bytes for uniqueness.
  """

  @doc "Generates a 32-char lowercase hex string (16 random bytes)."
  def generate do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  @doc "Generates a 16-char lowercase hex string (8 random bytes)."
  def generate_short do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  @doc "Generates a prefixed ID: prefix_<16 hex chars>."
  def generate(prefix) when is_binary(prefix) do
    prefix <> "_" <> generate_short()
  end
end
