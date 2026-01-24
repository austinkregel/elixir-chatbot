defmodule ChatBotWeb.TestController do
  use ChatBotWeb, :controller

  def test_learning(conn, %{"input" => input}) do
    {:ok, extracted_data} = ChatBot.Learner.learn_from_input("Echo", input)

    json(conn, %{
      status: "success",
      input: input,
      extracted_data: extracted_data
    })
  end

  def test_knowledge(conn, _params) do
    knowledge = ChatBot.KnowledgeStore.get_knowledge("Echo")

    json(conn, %{
      status: "success",
      knowledge: knowledge
    })
  end

  def add_test_knowledge(conn, _params) do
    # Add some test knowledge to demonstrate the display
    ChatBot.KnowledgeStore.add_person("Echo", "John", %{
      "age" => "30",
      "occupation" => "developer",
      "birthdate" => "1994-01-15",
      "favorite_holidays" => ["christmas", "new year"]
    })

    ChatBot.KnowledgeStore.add_pet("Echo", "Fluffy", %{
      "species" => "cat",
      "breed" => "persian",
      "age" => "3",
      "color" => "white"
    })

    ChatBot.KnowledgeStore.add_room("Echo", "living room", %{
      "type" => "living room",
      "size" => "large",
      "purpose" => "entertainment"
    })

    ChatBot.KnowledgeStore.add_device("Echo", "laptop", %{
      "type" => "computer",
      "brand" => "Apple",
      "model" => "MacBook Pro",
      "location" => "office"
    })

    knowledge = ChatBot.KnowledgeStore.get_knowledge("Echo")

    json(conn, %{
      status: "success",
      message: "Test knowledge added",
      knowledge: knowledge
    })
  end
end
