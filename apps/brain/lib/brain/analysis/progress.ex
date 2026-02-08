defmodule Brain.Analysis.Progress do
  @moduledoc "Lightweight progress reporting for the analysis pipeline.\n\nWhen `:progress` is present in opts (with `:conversation_id` and `:message_id`),\nthis module broadcasts LiveView-friendly events on the `brain:analysis` topic.\n"

  alias Phoenix.PubSub
  @topic "brain:analysis"

  @spec report(keyword(), atom(), map()) :: :ok
  def report(opts, step, data \\ %{}) when is_list(opts) and is_atom(step) and is_map(data) do
    progress = Keyword.get(opts, :progress)

    if is_map(progress) do
      conversation_id =
        Map.get(progress, :conversation_id) || Map.get(progress, "conversation_id")

      message_id = Map.get(progress, :message_id) || Map.get(progress, "message_id")

      if is_binary(conversation_id) and is_binary(message_id) do
        payload =
          data
          |> Map.put(:step, step)
          |> Map.put(:conversation_id, conversation_id)
          |> Map.put(:message_id, message_id)
          |> Map.put_new(:timestamp, System.system_time(:millisecond))

        PubSub.broadcast(Brain.PubSub, @topic, {:analysis_progress, payload})
      end
    end

    :ok
  rescue
    _ -> :ok
  end
end