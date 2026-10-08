defmodule ChatWeb.CoreComponentsTest do
  @moduledoc """
  The core components that changed with the design language: the flash edge,
  inline and small inputs, and the table's meaning washes and row actions.
  """
  use ExUnit.Case, async: true

  import Phoenix.LiveViewTest

  alias ChatWeb.CoreComponents

  defp slot(text), do: [%{__slot__: :inner_block, inner_block: fn _, _ -> text end}]

  describe "flash/1" do
    test "its whole edge is one hairline; no 4px left edge" do
      for kind <- [:info, :error] do
        html = render_component(&CoreComponents.flash/1, kind: kind, inner_block: slot("Saved"))

        refute html =~ "border-l-4"
        assert html =~ "border "
      end
    end

    test "the kind is carried by the icon and the edge color" do
      error = render_component(&CoreComponents.flash/1, kind: :error, inner_block: slot("Failed"))

      assert error =~ "border-red"
      assert error =~ "hero-exclamation-circle"
    end
  end

  describe "input/1 inline" do
    test "drops the wrapper margin and the visible label, keeping it as aria-label" do
      html =
        render_component(&CoreComponents.input/1,
          name: "token",
          label: "Token value",
          value: "",
          inline: true
        )

      refute html =~ "mb-space-sm"
      refute html =~ "text-label"
      assert html =~ ~s(aria-label="Token value")
    end

    test "still shows errors as a red border and caption" do
      html =
        render_component(&CoreComponents.input/1,
          name: "token",
          label: "Token value",
          value: "",
          inline: true,
          errors: ["must not be empty"]
        )

      assert html =~ "border-red"
      assert html =~ "must not be empty"
      assert html =~ "text-red"
    end

    test "an inline field without a label raises" do
      assert_raise ArgumentError, ~r/still needs its label/, fn ->
        render_component(&CoreComponents.input/1, name: "token", value: "", inline: true)
      end
    end

    test "inline applies to selects and textareas" do
      select =
        render_component(&CoreComponents.input/1,
          type: "select",
          name: "mode",
          label: "Mode",
          value: "a",
          options: [{"A", "a"}],
          inline: true
        )

      refute select =~ "mb-space-sm"
      assert select =~ ~s(aria-label="Mode")

      textarea =
        render_component(&CoreComponents.input/1,
          type: "textarea",
          name: "notes",
          label: "Notes",
          value: "",
          inline: true
        )

      refute textarea =~ "mb-space-sm"
      assert textarea =~ ~s(aria-label="Notes")
    end

    test "a field outside a row keeps its margin and visible label" do
      html = render_component(&CoreComponents.input/1, name: "world", label: "World name", value: "")

      assert html =~ "mb-space-sm"
      assert html =~ "World name"
      refute html =~ "aria-label"
    end
  end

  describe "input/1 size" do
    test ":sm is control-sm with dense body type; :md is the default" do
      small = render_component(&CoreComponents.input/1, name: "t", label: "T", value: "", size: :sm)
      assert small =~ "h-control-sm"
      assert small =~ "text-body-dense"

      regular = render_component(&CoreComponents.input/1, name: "t", label: "T", value: "")
      assert regular =~ "h-control-md"
    end

    test "an unknown size raises" do
      assert_raise ArgumentError, ~r/no treatment for size :xl/, fn ->
        render_component(&CoreComponents.input/1, name: "t", label: "T", value: "", size: :xl)
      end
    end

    test "a checkbox has one size" do
      assert_raise ArgumentError, ~r/a checkbox has one size/, fn ->
        render_component(&CoreComponents.input/1, type: "checkbox", name: "c", label: "C", value: true, size: :sm)
      end
    end
  end

  describe "table/1" do
    defp rows, do: [%{id: 1, origin: :default}, %{id: 2, origin: :computed}]

    defp col do
      [%{__slot__: :col, label: "Id", inner_block: fn _, row -> to_string(row.id) end}]
    end

    test "a row with a meaning wash takes it instead of the stripe" do
      html =
        render_component(&CoreComponents.table/1,
          id: "t",
          rows: rows(),
          col: col(),
          row_class: fn row -> if row.origin == :default, do: "bg-origin-standin-wash" end
        )

      [first, second] = Regex.scan(~r/<tr[^>]*class="([^"]*)"/, html, capture: :all_but_first)
      assert hd(first) =~ "bg-origin-standin-wash"
      refute hd(first) =~ "even:bg-surface-sunk"
      assert hd(second) =~ "even:bg-surface-sunk"
    end

    test "row actions sit space-xs apart" do
      action = [%{__slot__: :action, inner_block: fn _, _ -> "a" end}]

      html = render_component(&CoreComponents.table/1, id: "t", rows: rows(), col: col(), action: action)

      assert html =~ "gap-space-xs"
      refute html =~ "gap-space-lg"
    end
  end
end
