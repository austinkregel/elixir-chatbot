defmodule Brain.Response.PrimitiveTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Primitive
  alias Brain.Response.PrimitiveTypes

  describe "Primitive.new/3" do
    test "creates a primitive with type and variant" do
      p = Primitive.new(:acknowledgment, :social, %{speech_act_sub_type: :greeting})

      assert p.type == :acknowledgment
      assert p.variant == :social
      assert p.content.speech_act_sub_type == :greeting
      assert p.rendered == nil
      assert p.confidence == 1.0
    end

    test "creates a primitive with defaults" do
      p = Primitive.new(:hedging)

      assert p.type == :hedging
      assert p.variant == nil
      assert p.content == %{}
    end
  end

  describe "Primitive.render/2" do
    test "sets the rendered text" do
      p = Primitive.new(:acknowledgment, :social) |> Primitive.render("Hello!")

      assert p.rendered == "Hello!"
    end
  end

  describe "Primitive.put_content/3" do
    test "adds a key to the content map" do
      p = Primitive.new(:content, :factual)
          |> Primitive.put_content(:fact, "The sky is blue")

      assert p.content.fact == "The sky is blue"
    end
  end

  describe "Primitive.merge_content/2" do
    test "merges additional content" do
      p = Primitive.new(:content, :factual, %{fact: "original"})
          |> Primitive.merge_content(%{confidence: 0.9, source: :database})

      assert p.content.fact == "original"
      assert p.content.confidence == 0.9
      assert p.content.source == :database
    end
  end

  describe "Primitive.rendered?/1" do
    test "returns false when not rendered" do
      refute Primitive.rendered?(Primitive.new(:hedging))
    end

    test "returns true when rendered" do
      p = Primitive.new(:hedging) |> Primitive.render("I think")
      assert Primitive.rendered?(p)
    end
  end

  describe "Primitive.join_rendered/1" do
    test "joins rendered primitives with spaces" do
      primitives = [
        Primitive.new(:hedging) |> Primitive.render("I think"),
        Primitive.new(:content, :factual) |> Primitive.render("the weather is nice."),
        Primitive.new(:follow_up, :continuation) |> Primitive.render("Anything else?")
      ]

      result = Primitive.join_rendered(primitives)
      assert result == "I think the weather is nice. Anything else?"
    end

    test "skips unrendered primitives" do
      primitives = [
        Primitive.new(:hedging) |> Primitive.render("I think"),
        Primitive.new(:content, :factual),
        Primitive.new(:follow_up, :continuation) |> Primitive.render("Anything else?")
      ]

      result = Primitive.join_rendered(primitives)
      assert result == "I think Anything else?"
    end

    test "handles empty list" do
      assert Primitive.join_rendered([]) == ""
    end
  end

  describe "PrimitiveTypes.valid?/1" do
    test "validates a well-formed primitive" do
      p = Primitive.new(:acknowledgment, :social, %{speech_act_sub_type: :greeting})
      assert PrimitiveTypes.valid?(p)
    end

    test "validates hedging with confidence_level" do
      p = Primitive.new(:hedging, nil, %{confidence_level: 0.4})
      assert PrimitiveTypes.valid?(p)
    end

    test "rejects invalid variant" do
      p = Primitive.new(:acknowledgment, :nonexistent, %{})
      refute PrimitiveTypes.valid?(p)
    end

    test "rejects missing required content" do
      p = Primitive.new(:framing, :affirmative, %{})
      refute PrimitiveTypes.valid?(p)
    end

    test "validates with required content present" do
      p = Primitive.new(:framing, :affirmative, %{confirmed_fact: "The sky is blue"})
      assert PrimitiveTypes.valid?(p)
    end
  end

  describe "PrimitiveTypes.variants/1" do
    test "returns variants for each type" do
      assert :social in PrimitiveTypes.variants(:acknowledgment)
      assert :action in PrimitiveTypes.variants(:acknowledgment)
      assert :factual in PrimitiveTypes.variants(:content)
      assert :reflective in PrimitiveTypes.variants(:content)
      assert :empathy in PrimitiveTypes.variants(:attunement)
      assert :clarification in PrimitiveTypes.variants(:follow_up)
    end

    test "returns [nil] for types without variants" do
      assert PrimitiveTypes.variants(:hedging) == [nil]
      assert PrimitiveTypes.variants(:transition) == [nil]
    end
  end

  describe "PrimitiveTypes.all_types/0" do
    test "returns all 8 primitive types" do
      types = PrimitiveTypes.all_types()
      assert length(types) == 8
      assert :acknowledgment in types
      assert :framing in types
      assert :hedging in types
      assert :content in types
      assert :attunement in types
      assert :follow_up in types
      assert :contradiction_response in types
      assert :transition in types
    end
  end
end
