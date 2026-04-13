defmodule Atlas.Graph.AgtypeExtension do
  @moduledoc """
  Postgrex extension for Apache AGE's `agtype` data type.

  AGE returns graph elements (vertices, edges, paths) as `agtype`.
  This extension tells Postgrex to decode agtype values using text format,
  which gives us the human-readable JSON-like representation that our
  `Atlas.Graph.Cypher` module then parses into Elixir structs.
  """

  @behaviour Postgrex.Extension

  @impl true
  def init(_opts), do: []

  @impl true
  def matching(_state), do: [type: "agtype"]

  @impl true
  def format(_state), do: :text

  @impl true
  def encode(_state) do
    quote do
      bin when is_binary(bin) ->
        [<<byte_size(bin)::signed-32>> | bin]
    end
  end

  @impl true
  def decode(_state) do
    quote do
      <<len::signed-32, bin::binary-size(len)>> ->
        :binary.copy(bin)
    end
  end
end
