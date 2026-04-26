-- Bootstraps Apache AGE on a fresh Postgres data volume.
--
-- This file is mounted into /docker-entrypoint-initdb.d/ by docker-compose
-- and runs exactly once when the data directory is empty (i.e. after
-- `docker compose down -v && docker compose up`). It only initializes
-- the default POSTGRES_DB; other databases (such as the *_test database
-- created by Ecto for the test suite) must run `mix atlas.bootstrap_age`
-- explicitly.

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $$
DECLARE
  graph_name text;
BEGIN
  FOREACH graph_name IN ARRAY ARRAY[
    'knowledge_graph',
    'user_graph',
    'semantic_graph',
    'conversation_graph',
    'epistemic_graph',
    'pos_graph'
  ] LOOP
    IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = graph_name) THEN
      PERFORM ag_catalog.create_graph(graph_name);
    END IF;
  END LOOP;
END $$;
