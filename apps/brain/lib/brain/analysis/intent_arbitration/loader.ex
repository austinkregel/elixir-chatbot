defmodule Brain.Analysis.IntentArbitration.Loader do
  @moduledoc """
  Loads, at boot, each supported language's `Brain.Analysis.IntentArbitration`
  model, which `mix train_intent_arbitration` trains.

  It is required: every disagreement between the `:intent_full` and
  `:intent_domain` classifiers is resolved with it. A missing or unreadable
  artifact fails boot, naming the file and the task that writes it.

  The models are held in `:persistent_term`, so this child has no process of
  its own once they are loaded.
  """

  alias Brain.Analysis.IntentArbitration

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
      IntentArbitration.put_loaded(IntentArbitration.read!(language))
    end)
  end
end
