defmodule Brain.Response.Enricher do
  @moduledoc """
  Enriches responses with live data from external services.

  The Enricher integrates with the Service Dispatcher to fetch real-time
  data (weather, news, etc.) and inject it into response templates.

  ## Enrichment Flow

  1. Receives intent, slots, and context from the response generator
  2. Calls the Service Dispatcher to get enrichment data
  3. Merges enrichment data into the context
  4. Updates template slot values with enriched data
  5. Returns enriched response with metadata

  ## Usage

      # In response generator, after template selection:
      context = Enricher.prepare_context(intent, slots, context)

      # After getting a response:
      {:ok, enriched_response} = Enricher.enrich_response(response, context)

  ## Context Keys Added

  After calling `prepare_context/3`, the context will include:
  - `:enriched_data` - Map of field names to values from services
  - `:enrichment_status` - `:success`, `:failed`, or `:not_configured`
  - `:enrichment_service` - Name of the service that provided data
  - `:available_services` - List of configured services for this intent
  """

  require Logger

  alias Brain.Services.Dispatcher
  alias Brain.Response.TemplateStore

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Prepare context with enrichment data before template selection.

  This should be called early in the response generation pipeline,
  before template selection, so that templates can use `enriched:field`
  conditions.

  ## Parameters
    - intent: The classified intent
    - slots: Map of filled slots (e.g., %{location: "NYC"})
    - context: Current generation context

  ## Returns
    Updated context map with enrichment information.
  """
  @spec prepare_context(String.t(), map(), map()) :: map()
  def prepare_context(intent, slots, context) do
    # Check which services are available for this intent
    available_services = get_available_services(intent, context)

    enriched =
      if available_services == [] do
        context
        |> Map.put(:enrichment_status, :not_configured)
        |> Map.put(:enriched_data, %{})
        |> Map.put(:available_services, [])
      else
        case Dispatcher.dispatch(intent, slots, context) do
          {:ok, enrichment_data} ->
            context
            |> Map.put(:enriched_data, enrichment_data)
            |> Map.put(:enrichment_status, :success)
            |> Map.put(:enrichment_service, get_enrichment_service(intent))
            |> Map.put(:available_services, available_services)

          {:error, reason} ->
            Logger.debug("Enrichment failed",
              intent: intent,
              reason: inspect(reason)
            )

            context
            |> Map.put(:enriched_data, %{})
            |> Map.put(:enrichment_status, :failed)
            |> Map.put(:enrichment_error, reason)
            |> Map.put(:available_services, available_services)

          :no_handler ->
            context
            |> Map.put(:enrichment_status, :not_configured)
            |> Map.put(:enriched_data, %{})
            |> Map.put(:available_services, [])
        end
      end

    enrich_from_graph(enriched)
  end

  defp enrich_from_graph(context) do
    entities = Map.get(context, :entities, [])

    graph_enrichment = %{
      entity_context: Enum.map(entities, fn e ->
        %{
          entity: e,
          neighbors: Map.get(e, :graph_neighbors, []),
          node: if(Map.get(e, :graph_known, false), do: e, else: nil)
        }
      end),
      conversation_topics: Map.get(context, :conversation_topics, [])
    }

    Map.put(context, :graph_enrichment, graph_enrichment)
  end

  @doc """
  Enrich a response text by substituting enrichment placeholders.

  This substitutes placeholders like `$temperature`, `$conditions` with
  actual values from the enrichment data.

  ## Parameters
    - response: The response text (may contain placeholders)
    - context: Context with enrichment data

  ## Returns
    `{:ok, enriched_response}` — always succeeds
  """
  @spec enrich_response(String.t(), map()) :: {:ok, String.t()}
  def enrich_response(response, context) when is_binary(response) do
    enriched_data = Map.get(context, :enriched_data, %{})

    if enriched_data == %{} do
      {:ok, response}
    else
      enriched = substitute_enrichment_placeholders(response, enriched_data)
      {:ok, enriched}
    end
  end

  def enrich_response(response, _context), do: {:ok, response}

  @doc """
  Check if a response template can be enriched with available data.

  Returns true if all required enrichment fields are present.

  ## Parameters
    - template: Template struct or map with :requires_enrichment field
    - context: Context with enrichment data
  """
  @spec can_enrich?(map(), map()) :: boolean()
  def can_enrich?(template, context) do
    required_fields = get_required_enrichment(template)
    enriched_data = Map.get(context, :enriched_data, %{})

    Enum.all?(required_fields, fn field ->
      field_key = safe_field_atom(field)
      Map.has_key?(enriched_data, field_key) or Map.has_key?(enriched_data, to_string(field))
    end)
  end

  defp safe_field_atom(field) when is_atom(field), do: field
  defp safe_field_atom(field) when is_binary(field) do
    String.to_existing_atom(field)
  rescue
    ArgumentError -> field
  end

  @doc """
  Combine slot substitution with enrichment.

  This is a convenience function that performs both entity slot
  substitution and enrichment placeholder substitution.

  ## Parameters
    - template: Template text with placeholders
    - entities: List of entity maps
    - context: Context with enrichment data

  ## Returns
    Fully substituted response text.
  """
  @spec substitute_all(String.t(), list(), map()) :: String.t()
  def substitute_all(template, entities, context) do
    # First do entity/slot substitution
    response = TemplateStore.substitute_slots(template, entities)

    # Then do enrichment substitution
    enriched_data = Map.get(context, :enriched_data, %{})
    substitute_enrichment_placeholders(response, enriched_data)
  end

  @doc """
  Get enrichment status summary for response metadata.
  """
  @spec get_enrichment_metadata(map()) :: map()
  def get_enrichment_metadata(context) do
    %{
      status: Map.get(context, :enrichment_status, :not_configured),
      service: Map.get(context, :enrichment_service),
      error: Map.get(context, :enrichment_error),
      fields: Map.get(context, :enriched_data, %{}) |> Map.keys()
    }
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_available_services(intent, context) do
    world = Map.get(context, :world_id) || Map.get(context, :world, "default")

    Dispatcher.list_services(world: world)
    |> Enum.filter(fn service ->
      service.configured and intent in service.supported_intents
    end)
    |> Enum.map(& &1.name)
  end

  defp get_enrichment_service(intent) do
    case Dispatcher.find_service(intent) do
      nil -> nil
      service -> service.name()
    end
  end

  defp get_required_enrichment(%{requires_enrichment: fields}) when is_list(fields) do
    fields
  end

  defp get_required_enrichment(%{"requires_enrichment" => fields}) when is_list(fields) do
    fields
  end

  defp get_required_enrichment(_), do: []

  defp substitute_enrichment_placeholders(text, enriched_data) when is_binary(text) do
    Brain.Response.Formatting.substitute_placeholders(text, enriched_data)
  end

end
