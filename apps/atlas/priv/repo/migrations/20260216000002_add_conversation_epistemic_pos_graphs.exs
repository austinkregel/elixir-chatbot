defmodule Atlas.Repo.Migrations.AddConversationEpistemicPosGraphs do
  use Ecto.Migration

  @disable_ddl_transaction true

  def up do
    execute("""
    DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'conversation_graph')
      THEN PERFORM ag_catalog.create_graph('conversation_graph'); END IF;
      IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'epistemic_graph')
      THEN PERFORM ag_catalog.create_graph('epistemic_graph'); END IF;
      IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'pos_graph')
      THEN PERFORM ag_catalog.create_graph('pos_graph'); END IF;
    END $$;
    """)
  end

  def down do
    execute("SELECT ag_catalog.drop_graph('pos_graph', true)")
    execute("SELECT ag_catalog.drop_graph('epistemic_graph', true)")
    execute("SELECT ag_catalog.drop_graph('conversation_graph', true)")
  end
end
