defmodule FourthWall.IDTest do
  use ExUnit.Case, async: false

  alias FourthWall.ID

  describe "generate/0" do
    test "returns 32-char hex string" do
      id = ID.generate()
      assert String.length(id) == 32
      assert id =~ ~r/^[0-9a-f]{32}$/
    end
  end

  describe "generate_short/0" do
    test "returns 16-char hex string" do
      id = ID.generate_short()
      assert String.length(id) == 16
      assert id =~ ~r/^[0-9a-f]{16}$/
    end
  end

  describe "generate/1" do
    test "with prefix returns fact_ followed by 16 hex chars" do
      id = ID.generate("fact")
      assert String.starts_with?(id, "fact_")
      suffix = String.replace_prefix(id, "fact_", "")
      assert String.length(suffix) == 16
      assert suffix =~ ~r/^[0-9a-f]{16}$/
    end
  end

  describe "uniqueness" do
    test "generate 1000 IDs, all unique" do
      ids = for _ <- 1..1000, do: ID.generate()
      assert length(Enum.uniq(ids)) == 1000
    end
  end
end
