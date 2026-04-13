defmodule Atlas.Graph.TypesTest do
  use ExUnit.Case, async: false

  alias Atlas.Graph.Types.{Vertex, Edge, Path}

  describe "Vertex" do
    test "creates vertex with all fields" do
      v = %Vertex{id: 1, label: "Person", properties: %{"name" => "Alice"}}
      assert v.id == 1
      assert v.label == "Person"
      assert v.properties == %{"name" => "Alice"}
    end

    test "creates vertex with defaults" do
      v = %Vertex{}
      assert v.id == nil
      assert v.label == nil
      assert v.properties == nil
    end
  end

  describe "Edge" do
    test "creates edge with all fields" do
      e = %Edge{id: 1, start_id: 10, end_id: 20, label: "KNOWS", properties: %{"since" => 2020}}
      assert e.id == 1
      assert e.start_id == 10
      assert e.end_id == 20
      assert e.label == "KNOWS"
    end

    test "creates edge with defaults" do
      e = %Edge{}
      assert e.id == nil
      assert e.label == nil
    end
  end

  describe "Path" do
    test "creates path with vertices and edges" do
      v1 = %Vertex{id: 1, label: "Person", properties: %{}}
      v2 = %Vertex{id: 2, label: "Person", properties: %{}}
      e = %Edge{id: 1, start_id: 1, end_id: 2, label: "KNOWS", properties: %{}}

      p = %Path{vertices: [v1, v2], edges: [e]}
      assert length(p.vertices) == 2
      assert length(p.edges) == 1
    end

    test "creates empty path" do
      p = %Path{}
      assert p.vertices == []
      assert p.edges == []
    end
  end
end
