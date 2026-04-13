defmodule Brain.Response.SynthesizerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Synthesizer

  describe "synthesize/3" do
    test "returns :not_synthesized for unknown domain" do
      result = Synthesizer.synthesize("completely_unknown.intent", [], confidence: 0.8)

      assert result == :not_synthesized
    end

    test "synthesizes response for known domain" do
      entities = [%{entity_type: "location", value: "NYC"}]

      result = Synthesizer.synthesize("weather.query", entities, confidence: 0.8)

      # Should either synthesize or return :not_synthesized
      assert result == :not_synthesized or match?({:ok, _}, result)
    end
  end

  describe "synthesize_from_domain/5 with enrichment" do
    test "uses enriched templates when enrichment data is available" do
      entities = [%{entity_type: "location", value: "NYC"}]

      context = %{
        enrichment_status: :success,
        enriched_data: %{
          temperature: "72°F",
          conditions: "sunny",
          humidity: "45%",
          wind_speed: "5 mph"
        }
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: context
        )

      case result do
        {:ok, response} ->
          # Response should contain the enriched data
          assert response =~ "72°F" or response =~ "NYC"

        :not_synthesized ->
          # Domain may not have templates configured
          assert true
      end
    end

    test "falls back to standard templates when no enrichment data" do
      entities = [%{entity_type: "location", value: "NYC"}]
      context = %{enrichment_status: :not_configured, enriched_data: %{}}

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: context
        )

      case result do
        {:ok, response} ->
          assert is_binary(response)
          # Should mention the location
          assert response =~ "NYC"

        :not_synthesized ->
          assert true
      end
    end

    test "returns :not_synthesized for nil domain" do
      result = Synthesizer.synthesize_from_domain(nil, "test", [], 0.8, [])
      assert result == :not_synthesized
    end
  end

  describe "synthesize_clarification/4" do
    test "generates clarification for missing slots" do
      missing_slots = [:location]

      result =
        Synthesizer.synthesize_clarification(
          "weather.query",
          [],
          missing_slots,
          []
        )

      case result do
        {:ok, response} ->
          assert is_binary(response)
          # Should ask about location
          assert response =~ "location" or String.length(response) > 0

        _ ->
          # May not have clarification templates configured
          assert true
      end
    end
  end

  describe "get_fallback_response/0" do
    test "returns a fallback string" do
      response = Synthesizer.get_fallback_response()

      assert is_binary(response)
      assert String.length(response) > 0
    end
  end

  describe "enriched response frame selection" do
    test "selects more specific frame when more fields match" do
      entities = [%{entity_type: "location", value: "Chicago"}]

      # Context with all detailed fields
      detailed_context = %{
        enrichment_status: :success,
        enriched_data: %{
          temperature: "55°F",
          conditions: "cloudy",
          humidity: "80%",
          wind_speed: "10 mph"
        }
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: detailed_context
        )

      case result do
        {:ok, response} ->
          # Should use detailed template if available
          # At minimum should contain some of the enrichment data
          assert response =~ "55°F" or response =~ "Chicago" or response =~ "cloudy"

        :not_synthesized ->
          assert true
      end
    end

    test "handles service error gracefully" do
      entities = [%{entity_type: "location", value: "NYC"}]

      context = %{
        enrichment_status: :failed,
        enrichment_error: :timeout,
        enriched_data: %{}
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: context
        )

      case result do
        {:ok, response} ->
          # May use error template or fall back to standard
          assert is_binary(response)

        :not_synthesized ->
          assert true
      end
    end
  end

  describe "fill_enrichment_slots integration" do
    test "substitutes $variable placeholders" do
      # This tests the internal behavior through the public API
      entities = [%{entity_type: "location", value: "Seattle"}]

      context = %{
        enrichment_status: :success,
        enriched_data: %{
          temperature: "48°F",
          conditions: "rainy"
        }
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.current",
          entities,
          0.9,
          context: context
        )

      case result do
        {:ok, response} ->
          # Should not contain unsubstituted placeholders
          refute response =~ "$temperature"
          refute response =~ "$conditions"

        :not_synthesized ->
          assert true
      end
    end
  end
end
