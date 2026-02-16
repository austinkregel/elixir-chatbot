defmodule Atlas.Repo.Migrations.SetupApacheAge do
  use Ecto.Migration

  def up do
    execute("CREATE EXTENSION IF NOT EXISTS age")
    execute("LOAD 'age'")
    execute("SET search_path = ag_catalog, \"$user\", public")
    execute("SELECT create_graph('knowledge_graph')")
    execute("SELECT create_graph('user_graph')")
    execute("SELECT create_graph('semantic_graph')")
    # Reset search_path so subsequent migrations create tables in public schema
    execute("SET search_path = public")
  end

  def down do
    execute("SELECT drop_graph('semantic_graph', true)")
    execute("SELECT drop_graph('user_graph', true)")
    execute("SELECT drop_graph('knowledge_graph', true)")
    execute("DROP EXTENSION IF EXISTS age CASCADE")
  end
end
