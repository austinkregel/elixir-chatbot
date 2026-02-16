defmodule Brain.Response.EnricherTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Response.Enricher

  describe "prepare_context/3" do
    test "returns context with enrichment_status :not_configured when no handler" do
      context = %{world_id: "default"}

      result = Enricher.prepare_context("unknown.intent", %{}, context)

      assert result.enrichment_status == :not_configured
      assert result.enriched_data == %{}
      assert result.available_services == []
    end

    test "preserves existing context keys" do
      context = %{
        world_id: "default",
        entities: [%{entity_type: "location", value: "NYC"}],
        custom_key: "preserved"
      }

      result = Enricher.prepare_context("unknown.intent", %{}, context)

      assert result.entities == context.entities
      assert result.custom_key == "preserved"
    end
  end

  describe "enrich_response/2" do
    test "substitutes enrichment placeholders in response text" do
      response = "It's currently $temperature and $conditions in NYC."

      context = %{
        enriched_data: %{
          temperature: "72°F",
          conditions: "sunny"
        }
      }

      {:ok, enriched} = Enricher.enrich_response(response, context)

      assert enriched == "It's currently 72°F and sunny in NYC."
    end

    test "handles @ placeholder format" do
      response = "The weather is @temperature with @conditions."

      context = %{
        enriched_data: %{
          temperature: "65°F",
          conditions: "partly cloudy"
        }
      }

      {:ok, enriched} = Enricher.enrich_response(response, context)

      assert enriched == "The weather is 65°F with partly cloudy."
    end

    test "returns original response when no enrichment data" do
      response = "Let me check the weather for you."
      context = %{enriched_data: %{}}

      {:ok, result} = Enricher.enrich_response(response, context)

      assert result == response
    end

    test "returns original response when enriched_data missing from context" do
      response = "Let me check the weather for you."
      context = %{}

      {:ok, result} = Enricher.enrich_response(response, context)

      assert result == response
    end

    test "skips nested map values in enrichment data" do
      response = "Temperature is $temperature. Raw is $raw."

      context = %{
        enriched_data: %{
          temperature: "72°F",
          raw: %{temp: 72, humidity: 45}
        }
      }

      {:ok, enriched} = Enricher.enrich_response(response, context)

      # $temperature replaced, $raw not replaced (it's a map)
      assert enriched =~ "Temperature is 72°F"
      assert enriched =~ "$raw"
    end

    test "handles numeric values" do
      response = "Humidity is $humidity percent."

      context = %{
        enriched_data: %{humidity: 45}
      }

      {:ok, enriched} = Enricher.enrich_response(response, context)

      assert enriched == "Humidity is 45 percent."
    end
  end

  describe "can_enrich?/2" do
    test "returns true when all required fields are present" do
      template = %{
        requires_enrichment: ["temperature", "conditions"]
      }

      context = %{
        enriched_data: %{temperature: "72°F", conditions: "sunny", humidity: "45%"}
      }

      assert Enricher.can_enrich?(template, context) == true
    end

    test "returns false when required fields are missing" do
      template = %{
        requires_enrichment: ["temperature", "conditions", "humidity"]
      }

      context = %{
        enriched_data: %{temperature: "72°F"}
      }

      assert Enricher.can_enrich?(template, context) == false
    end

    test "returns true when no enrichment required" do
      template = %{text: "Simple template"}
      context = %{enriched_data: %{}}

      assert Enricher.can_enrich?(template, context) == true
    end

    test "handles string keys in requires_enrichment" do
      template = %{
        "requires_enrichment" => ["temperature"]
      }

      context = %{
        enriched_data: %{temperature: "72°F"}
      }

      assert Enricher.can_enrich?(template, context) == true
    end
  end

  describe "substitute_all/3" do
    test "combines entity slots and enrichment data" do
      template = "It's $temperature in $location."

      entities = [
        %{entity_type: "location", value: "NYC"}
      ]

      context = %{
        enriched_data: %{temperature: "72°F"}
      }

      result = Enricher.substitute_all(template, entities, context)

      assert result == "It's 72°F in NYC."
    end
  end

  describe "get_enrichment_metadata/1" do
    test "extracts enrichment metadata from context" do
      context = %{
        enrichment_status: :success,
        enrichment_service: :weather,
        enriched_data: %{temperature: "72°F", conditions: "sunny"}
      }

      metadata = Enricher.get_enrichment_metadata(context)

      assert metadata.status == :success
      assert metadata.service == :weather
      assert :temperature in metadata.fields
      assert :conditions in metadata.fields
    end

    test "handles missing enrichment data" do
      context = %{}

      metadata = Enricher.get_enrichment_metadata(context)

      assert metadata.status == :not_configured
      assert metadata.service == nil
      assert metadata.fields == []
    end

    test "includes error reason when present" do
      context = %{
        enrichment_status: :failed,
        enrichment_error: :rate_limited
      }

      metadata = Enricher.get_enrichment_metadata(context)

      assert metadata.status == :failed
      assert metadata.error == :rate_limited
    end
  end
end
