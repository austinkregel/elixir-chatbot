defmodule Brain.Analysis.SlotTypePreferences.Loader do
  @moduledoc """
  Loads, at boot, each supported language's `Brain.Analysis.SlotTypePreferences`
  model, which `mix train_slot_type_preferences` trains.

  It is required: the entity types an intent expects are weighed with it, and
  every gazetteer reading is scored with its preceding-tag likelihoods. A
  missing or unreadable artifact fails boot, naming the file and the task that
  writes it, and so does a model that does not cover every type the slot
  schemas list.

  The models are held in `:persistent_term`, so this child has no process of
  its own once they are loaded.
  """

  alias Brain.Analysis.SlotTypePreferences

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
    listed = SlotTypePreferences.listed_types()

    Enum.each(Brain.Language.supported(), fn language ->
      model = SlotTypePreferences.read!(language)
      SlotTypePreferences.check_listed!(model, listed)
      SlotTypePreferences.put_loaded(model)
    end)
  end
end
