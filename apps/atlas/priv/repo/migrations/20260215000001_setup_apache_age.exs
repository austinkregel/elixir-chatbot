defmodule Atlas.Repo.Migrations.SetupApacheAge do
  use Ecto.Migration

  @disable_ddl_transaction true

  def up do
    execute("CREATE EXTENSION IF NOT EXISTS age")

    execute("""
    DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'knowledge_graph')
      THEN PERFORM ag_catalog.create_graph('knowledge_graph'); END IF;
      IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'user_graph')
      THEN PERFORM ag_catalog.create_graph('user_graph'); END IF;
      IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'semantic_graph')
      THEN PERFORM ag_catalog.create_graph('semantic_graph'); END IF;
    END $$;
    """)
  end

  def down do
    execute("SELECT ag_catalog.drop_graph('semantic_graph', true)")
    execute("SELECT ag_catalog.drop_graph('user_graph', true)")
    execute("SELECT ag_catalog.drop_graph('knowledge_graph', true)")
    execute("DROP EXTENSION IF EXISTS age CASCADE")
  end
end
