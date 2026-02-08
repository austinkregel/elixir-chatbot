defmodule World do
  @moduledoc """
  World app - Training Worlds and Entity Discovery.

  This app manages isolated training environments (worlds) that provide:
  - World-scoped learning and embeddings
  - Entity discovery and type inference
  - Document ingestion
  - Training metrics and events
  """

  @doc """
  Returns the path to a file or directory within the World app's priv directory.

  This uses `Application.app_dir/2` to get the correct path regardless of
  whether the app is running from source or as a release.

  ## Examples

      World.priv_path("training_worlds")
      #=> "/path/to/apps/world/priv/training_worlds"
  """
  def priv_path(path) do
    case :code.priv_dir(:world) do
      {:error, _} ->
        # Fallback for when app isn't fully started
        Path.join(["apps", "world", "priv", path])

      priv_dir ->
        Path.join(priv_dir, path)
    end
  end
end
