defmodule FourthWallTest do
  use ExUnit.Case, async: false

  describe "version/0" do
    test "returns the version string" do
      assert FourthWall.version() == "0.1.0"
    end
  end
end
