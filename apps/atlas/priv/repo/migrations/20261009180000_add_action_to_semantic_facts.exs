defmodule Atlas.Repo.Migrations.AddActionToSemanticFacts do
  use Ecto.Migration

  # Consolidation stored a semantic fact as one string, "Action: <action> |
  # Examples: <examples>", so the fact's text was wrapped in boilerplate. The
  # action now has its own column and `content` holds the examples alone.
  #
  # Every row whose content is in that form is split at its first
  # " | Examples: ": the text between "Action: " and the separator becomes the
  # action, the text after it the content. A row whose content is not in that
  # form is the fact's own text: it is kept as it is, with no action. No row is
  # deleted.
  def up do
    alter table(:atlas_semantic_facts) do
      add :action, :string
    end

    execute(split_legacy_content_sql())
  end

  # The action goes back into the content, in the form it was stored in.
  def down do
    execute("""
    UPDATE atlas_semantic_facts
    SET content = 'Action: ' || action || ' | Examples: ' || content
    WHERE action IS NOT NULL
    """)

    alter table(:atlas_semantic_facts) do
      remove :action
    end
  end

  # Public so the conversion can be run against rows a test writes in the old
  # form. "Action: " is 8 characters, so the action starts at character 9 and
  # the separator can start no earlier; " | Examples: " is 13 characters. The
  # same split as `Atlas.Schemas.SemanticFact.split_legacy_content/1`.
  def split_legacy_content_sql do
    """
    UPDATE atlas_semantic_facts
    SET action = substring(content FROM 9 FOR position(' | Examples: ' IN substring(content FROM 9)) - 1),
        content = substring(content FROM 8 + position(' | Examples: ' IN substring(content FROM 9)) + 13)
    WHERE action IS NULL
      AND content LIKE 'Action: %'
      AND position(' | Examples: ' IN substring(content FROM 9)) > 0
    """
  end
end
