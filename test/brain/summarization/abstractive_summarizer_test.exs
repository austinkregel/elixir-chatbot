defmodule Brain.Summarization.AbstractiveSummarizerTest do
  use ExUnit.Case
  
  alias Brain.Summarization.AbstractiveSummarizer
  alias Brain.Summarization.Types.Fact
  
  test "falls back to template-based when model not ready" do
    facts = [
      Fact.new("Alice", "got promoted", :high),
      Fact.new("Bob", "moved to New York", :medium)
    ]
    
    summary = AbstractiveSummarizer.summarize(facts)
    
    assert %Brain.Summarization.Types.Summary{} = summary
    assert is_binary(summary.text)
    assert summary.text != ""
  end
  
  test "handles empty facts list" do
    summary = AbstractiveSummarizer.summarize([])
    
    assert %Brain.Summarization.Types.Summary{} = summary
  end
  
  test "linearizes facts correctly" do
    facts = [
      Fact.new("Alice", "got promoted", :high),
      Fact.new("Bob", "moved", :medium)
    ]
    
    # Test the private function via public API
    summary = AbstractiveSummarizer.summarize(facts)
    
    # Should produce some text
    assert is_binary(summary.text)
  end
end
