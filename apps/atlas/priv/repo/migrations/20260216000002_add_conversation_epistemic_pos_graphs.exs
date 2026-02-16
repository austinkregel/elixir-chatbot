defmodule Atlas.Repo.Migrations.AddConversationEpistemicPosGraphs do
  use Ecto.Migration

  def up do
    execute("LOAD 'age'")
    execute("SET search_path = ag_catalog, \"$user\", public")
    execute("SELECT create_graph('conversation_graph')")
    execute("SELECT create_graph('epistemic_graph')")
    execute("SELECT create_graph('pos_graph')")
    execute("SET search_path = public")
  end

  def down do
    execute("LOAD 'age'")
    execute("SET search_path = ag_catalog, \"$user\", public")
    execute("SELECT drop_graph('pos_graph', true)")
    execute("SELECT drop_graph('epistemic_graph', true)")
    execute("SELECT drop_graph('conversation_graph', true)")
    execute("SET search_path = public")
  end
end
