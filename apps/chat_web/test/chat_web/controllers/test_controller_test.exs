defmodule ChatWeb.TestControllerTest do
  use ChatWeb.ConnCase, async: false
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()

    # Start required GenServers for API endpoints
    ensure_started(Brain.KnowledgeStore)
    ensure_started(Brain.MemoryStore)
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.Analysis.LearningStore)

    :ok
  end

  describe "POST /api/test-learning" do
    test "returns extracted data from input", %{conn: conn} do
      conn = post(conn, "/api/test-learning", %{input: "My name is John"})

      response = json_response(conn, 200)
      assert response["status"] == "success"
      assert response["input"] == "My name is John"
      # extracted_data is a map with entities, facts, relationships, context
      assert is_map(response["extracted_data"])
    end

    test "extracts entities from input", %{conn: conn} do
      conn = post(conn, "/api/test-learning", %{input: "My name is John"})

      response = json_response(conn, 200)
      extracted = response["extracted_data"]
      assert is_list(extracted["entities"])
    end

    test "handles location input", %{conn: conn} do
      conn = post(conn, "/api/test-learning", %{input: "I live in New York"})

      response = json_response(conn, 200)
      assert response["status"] == "success"
      assert response["input"] == "I live in New York"
    end
  end

  describe "GET /api/test-knowledge" do
    test "returns knowledge for Echo persona", %{conn: conn} do
      conn = get(conn, "/api/test-knowledge")

      response = json_response(conn, 200)
      assert response["status"] == "success"
      assert is_map(response["knowledge"])
    end

    test "returns map structure for knowledge", %{conn: conn} do
      conn = get(conn, "/api/test-knowledge")

      response = json_response(conn, 200)
      assert response["status"] == "success"
      # Knowledge should be a map (possibly empty)
      assert is_map(response["knowledge"])
    end
  end

  describe "POST /api/add-test-knowledge" do
    test "adds test knowledge and returns updated state", %{conn: conn} do
      conn = post(conn, "/api/add-test-knowledge")

      response = json_response(conn, 200)
      assert response["status"] == "success"
      assert response["message"] == "Test knowledge added"
      assert is_map(response["knowledge"])

      # Verify specific knowledge was added
      knowledge = response["knowledge"]
      assert Map.has_key?(knowledge, "people") or Map.has_key?(knowledge, "pets")
    end

    test "can retrieve added knowledge via GET endpoint", %{conn: conn} do
      # Add knowledge first
      post(conn, "/api/add-test-knowledge")

      # Retrieve it
      conn = get(conn, "/api/test-knowledge")

      response = json_response(conn, 200)
      assert response["status"] == "success"

      knowledge = response["knowledge"]
      # Should have at least one of the categories added
      has_content =
        Map.has_key?(knowledge, "people") or
          Map.has_key?(knowledge, "pets") or
          Map.has_key?(knowledge, "rooms") or
          Map.has_key?(knowledge, "devices")

      assert has_content
    end
  end
end
