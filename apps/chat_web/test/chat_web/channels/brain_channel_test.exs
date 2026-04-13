defmodule ChatWeb.BrainChannelTest do
  use ChatWeb.ChannelCase, async: false
  import Brain.TestHelpers

  alias ChatWeb.BrainChannel

  setup do
    ensure_pubsub_started()

    # Start required services for Brain
    ensure_started(Brain.KnowledgeStore)
    ensure_started(Brain.MemoryStore)
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.Analysis.LearningStore)

    :ok
  end

  describe "join/3" do
    test "can join brain:status channel" do
      assert {:ok, _, socket} =
               socket(ChatWeb.UserSocket, "user_id", %{})
               |> subscribe_and_join(BrainChannel, "brain:status")

      leave(socket)
    end

    test "can join brain:learning channel" do
      assert {:ok, _, socket} =
               socket(ChatWeb.UserSocket, "user_id", %{})
               |> subscribe_and_join(BrainChannel, "brain:learning")

      leave(socket)
    end

    test "can join brain:conversations channel" do
      assert {:ok, _, socket} =
               socket(ChatWeb.UserSocket, "user_id", %{})
               |> subscribe_and_join(BrainChannel, "brain:conversations")

      leave(socket)
    end

    test "can join urgent:interrupt channel" do
      assert {:ok, _, socket} =
               socket(ChatWeb.UserSocket, "user_id", %{})
               |> subscribe_and_join(BrainChannel, "urgent:interrupt")

      leave(socket)
    end

    test "can join urgent:emergency channel" do
      assert {:ok, _, socket} =
               socket(ChatWeb.UserSocket, "user_id", %{})
               |> subscribe_and_join(BrainChannel, "urgent:emergency")

      leave(socket)
    end

    test "rejects unauthorized channel" do
      assert {:error, %{reason: "unauthorized"}} =
               socket(ChatWeb.UserSocket, "user_id", %{})
               |> subscribe_and_join(BrainChannel, "brain:unknown")
    end
  end

  describe "handle_in/3 - unknown events" do
    test "handles unknown events gracefully" do
      {:ok, _, socket} =
        socket(ChatWeb.UserSocket, "user_id", %{})
        |> subscribe_and_join(BrainChannel, "brain:status")

      ref = push(socket, "unknown_event", %{})
      refute_reply ref, :ok

      leave(socket)
    end
  end

  describe "terminate/2" do
    test "handles disconnect gracefully" do
      {:ok, _, socket} =
        socket(ChatWeb.UserSocket, "user_id", %{})
        |> subscribe_and_join(BrainChannel, "brain:status")

      # Leaving triggers terminate
      leave(socket)
      # No crash means success
    end
  end
end
