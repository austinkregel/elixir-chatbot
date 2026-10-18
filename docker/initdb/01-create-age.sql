-- Bootstraps Apache AGE on a fresh Postgres data volume.
--
-- This file is mounted into /docker-entrypoint-initdb.d/ by docker-compose
-- and runs exactly once when the data directory is empty (i.e. after
-- `docker compose down -v && docker compose up`). It only initializes
-- the default POSTGRES_DB; other databases (such as the *_test database
-- created by Ecto for the test suite) must run `mix atlas.bootstrap_age`
-- explicitly.
--
-- Creates the same graphs as `mix atlas.bootstrap_age`: the shared
-- `pos_graph`, and the "default" world's graph of each per-world kind, named
-- `<kind>__<world id>` by `Atlas.Graph.world_graph/2`. Every other world's
-- graphs are created when the world is.

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $$
DECLARE
  graph_name text;
BEGIN
  FOREACH graph_name IN ARRAY ARRAY[
    'pos_graph',
    'knowledge_graph__default',
    'user_graph__default',
    'semantic_graph__default',
    'conversation_graph__default',
    'epistemic_graph__default'
  ] LOOP
    IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = graph_name) THEN
      PERFORM ag_catalog.create_graph(graph_name);
    END IF;
  END LOOP;
END $$;
