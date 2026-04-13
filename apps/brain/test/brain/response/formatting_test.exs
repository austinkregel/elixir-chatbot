defmodule Brain.Response.FormattingTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Formatting

  describe "format_value/1" do
    test "formats strings as-is" do
      assert Formatting.format_value("hello") == "hello"
    end

    test "formats atoms to strings" do
      assert Formatting.format_value(:hello) == "hello"
    end

    test "formats numbers to strings" do
      assert Formatting.format_value(42) == "42"
      assert Formatting.format_value(3.14) == "3.14"
    end

    test "formats lists as comma-separated" do
      assert Formatting.format_value(["a", "b", "c"]) == "a, b, c"
    end

    test "formats other types with inspect" do
      result = Formatting.format_value(%{key: "val"})
      assert is_binary(result)
      assert result =~ "key"
    end
  end

  describe "substitute_placeholders/2" do
    test "replaces dollar-sign placeholders" do
      result = Formatting.substitute_placeholders("Hello $name!", %{name: "World"})
      assert result == "Hello World!"
    end

    test "replaces at-sign placeholders" do
      result = Formatting.substitute_placeholders("Hello @name!", %{name: "World"})
      assert result == "Hello World!"
    end

    test "skips nested map values" do
      data = %{name: "World", raw: %{nested: true}}
      result = Formatting.substitute_placeholders("Hello $name!", data)
      assert result == "Hello World!"
    end

    test "handles multiple replacements" do
      data = %{city: "NYC", temp: 72}
      result = Formatting.substitute_placeholders("$city is $temp degrees", data)
      assert result == "NYC is 72 degrees"
    end
  end
end
