defmodule Brain.ML.SpanReliability.Loader do
  @moduledoc """
  Loads, at boot, each supported language's `Brain.ML.SpanReliability` model,
  which `mix train_span_reliability` trains.

  It is required: every entity extraction decides its spans with it. A missing
  or unreadable artifact fails boot, naming the file and the task that writes
  it.

  The models are held in `:persistent_term`, so this child has no process of
  its own once they are loaded.
  """

  alias Brain.ML.SpanReliability

  def child_spec(_opts) do
    %{id: __MODULE__, start: {__MODULE__, :start_link, []}, restart: :temporary}
  end

  @doc false
  def start_link do
    load!()
    :ignore
  end

  @doc "Reads and installs every supported language's model."
  @spec load!() :: :ok
  def load! do
    Enum.each(Brain.Language.supported(), fn language ->
      SpanReliability.put_loaded(SpanReliability.read!(language))
    end)
  end
end
