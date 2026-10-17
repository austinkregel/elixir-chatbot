defmodule Brain.Analysis.FreeTextCapture.Loader do
  @moduledoc """
  Loads, at boot, each supported language's `Brain.Analysis.FreeTextCapture`
  model, which `mix train_free_text_capture` trains.

  It is required: `Brain.Analysis.SlotDetector` captures every free-text slot
  (a reminder's text) with it. A missing or unreadable artifact fails boot,
  naming the file and the task that writes it, and so does a model that
  captures a slot for an intent the intent registry does not declare it for.

  The models are held in `:persistent_term`, so this child has no process of
  its own once they are loaded.
  """

  alias Brain.Analysis.{FreeTextCapture, SlotDetector}

  def child_spec(_opts) do
    %{id: __MODULE__, start: {__MODULE__, :start_link, []}, restart: :temporary}
  end

  @doc false
  def start_link do
    load!()
    :ignore
  end

  @doc "Reads, checks and installs every supported language's model."
  @spec load!() :: :ok
  def load! do
    registry = SlotDetector.registry()

    Enum.each(Brain.Language.supported(), fn language ->
      model = FreeTextCapture.read!(language)
      FreeTextCapture.check_declared!(model, registry)
      FreeTextCapture.put_loaded(model)
    end)
  end
end
