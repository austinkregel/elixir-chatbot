defmodule Brain.Response.SurfaceRealizerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.{Primitive, SurfaceRealizer}

  describe "realize/2 with empty list" do
    test "raises with :empty_primitive_list reason" do
      error = assert_raise RuntimeError, fn -> SurfaceRealizer.realize([]) end

      assert error.message =~ "empty_primitive_list",
        "Expected :empty_primitive_list in error, got: #{error.message}"
    end
  end

  describe "realize/2 success path" do
    test "single primitive returns {rendered_list, response_text}" do
      p = Primitive.new(:acknowledgment, :social, %{speech_act_sub_type: :greeting})

      {rendered, response} = SurfaceRealizer.realize([p])

      assert is_binary(response), "Expected response to be a string, got: #{inspect(response)}"
      assert response != "", "Response should not be empty"

      assert is_list(rendered), "Expected rendered to be a list"
      assert length(rendered) == 1, "Should return one rendered primitive per input"

      [r] = rendered
      assert r.source == :ouro, "Rendered primitive should have source :ouro, got: #{inspect(r.source)}"
      assert Primitive.rendered?(r), "Primitive should be marked as rendered"
      assert r.rendered == response, "Rendered text should match the full response"
    end

    test "multi-primitive plan returns all primitives rendered with :ouro source" do
      primitives = [
        Primitive.new(:attunement, :empathy, %{sentiment_label: :negative}),
        Primitive.new(:content, :reflective, %{understood_meaning: "you had a tough day"}),
        Primitive.new(:follow_up, :elaboration, %{topic: "feelings"})
      ]

      {rendered, response} = SurfaceRealizer.realize(primitives)

      assert is_binary(response)
      assert response != ""

      assert length(rendered) == length(primitives),
        "Should return one rendered primitive per input (expected #{length(primitives)}, got #{length(rendered)})"

      for r <- rendered do
        assert r.source == :ouro, "Each rendered primitive should have source :ouro"
        assert Primitive.rendered?(r), "Each primitive should be marked as rendered"
      end
    end

    test "preserves primitive type and variant through realization" do
      p = Primitive.new(:hedging, :epistemic, %{confidence_level: 0.3})

      {[rendered], _response} = SurfaceRealizer.realize([p])

      assert rendered.type == :hedging, "Type should be preserved, got: #{inspect(rendered.type)}"
      assert rendered.variant == :epistemic, "Variant should be preserved, got: #{inspect(rendered.variant)}"
    end
  end

  describe "realize/2 opts forwarding" do
    test "passes analysis from opts through to OuroRealizer" do
      p = Primitive.new(:follow_up, :clarification, %{
        missing_slots: ["location"],
        ambiguity_type: :missing_slots,
        intent: "weather.query"
      })

      analysis = %Brain.Analysis.ChunkAnalysis{
        text: "What's the weather?",
        intent: "weather.query"
      }

      {rendered, response} = SurfaceRealizer.realize([p], analysis: analysis)

      assert is_binary(response)
      assert response != ""
      assert length(rendered) == 1
      assert hd(rendered).source == :ouro
    end

    test "works with empty opts" do
      p = Primitive.new(:content, :reflective, %{understood_meaning: "testing"})

      {rendered, response} = SurfaceRealizer.realize([p], [])

      assert is_binary(response)
      assert response != ""
      assert length(rendered) == 1
    end
  end
end
