defmodule Brain.Analysis.LexicalEvidence.Loader do
  @moduledoc """
  Loads, at boot, each supported language's `Brain.Analysis.LexicalEvidence`
  model and `Brain.Analysis.SpeechActTable`, which are trained together by
  `mix gen_micro_data`.

  Both are required: the feature vector has a lexical-evidence group, and the
  speech act of a classified intent comes from the table. A missing or
  unreadable artifact fails boot, naming the file and the task that writes it.

  The models and tables are held in `:persistent_term`, so this child has no
  process of its own once they are loaded.
  """

  alias Brain.Analysis.{LexicalEvidence, SpeechActTable}

  def child_spec(_opts) do
    %{id: __MODULE__, start: {__MODULE__, :start_link, []}, restart: :temporary}
  end

  @doc false
  def start_link do
    load!()
    :ignore
  end

  @doc "Reads and installs every supported language's model and table."
  @spec load!() :: :ok
  def load! do
    Enum.each(Brain.Language.supported(), fn language ->
      LexicalEvidence.put_loaded(LexicalEvidence.read!(language))
      SpeechActTable.put_loaded(SpeechActTable.read!(language))
    end)
  end
end
