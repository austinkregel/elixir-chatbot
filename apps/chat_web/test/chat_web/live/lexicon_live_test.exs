defmodule ChatWeb.LexiconLiveTest do
  @moduledoc """
  The lexicon isolation page: a human can look up a word and see what the brain
  knows, and where each answer came from.
  """
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest

  describe "mount" do
    test "renders the lookup form with no word entered", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/lexicon")

      assert html =~ "Lexicon"
      assert html =~ "Look up a word"
      refute html =~ "Owned facts"
    end
  end

  describe "looking up a word WordNet knows" do
    test "shows senses, definition and parts of speech", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> form("#lexicon-lookup", %{"word" => "dog"}) |> render_submit()

      assert html =~ "known"
      assert html =~ "noun"
      assert html =~ "domesticated by man since prehistoric times"
      assert html =~ "Senses ("
    end

    test "an example button performs the lookup", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> element("button[phx-value-word='dog']") |> render_click()

      assert html =~ "noun_animal"
    end
  end

  describe "negation" do
    test "a seeded morphological negator shows its root, affix and origin", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> form("#lexicon-lookup", %{"word" => "unable"}) |> render_submit()

      assert html =~ "Negation"
      assert html =~ "seed:wordnet"
      assert html =~ "seeded"
      # unable = un- + able
      assert html =~ "un-"
      assert html =~ "able"
    end

    test "a closed-class negator is reported without an owned fact", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> form("#lexicon-lookup", %{"word" => "never"}) |> render_submit()

      assert html =~ "Negation"
      # "never" negates, but WordNet holds no function words so nothing is seeded.
      assert html =~ "Closed class"
    end

    test "a word that does not negate says so", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> form("#lexicon-lookup", %{"word" => "cold"}) |> render_submit()

      # "cold" is the antonym of "hot" but is not derived from it by an affix.
      assert html =~ "No negation fact"
    end
  end

  describe "a word the brain does not know" do
    test "reports it as out of vocabulary with nothing owned", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> form("#lexicon-lookup", %{"word" => "zorblwidget"}) |> render_submit()

      assert html =~ "out of vocabulary"
      assert html =~ "Nothing stored"
    end
  end

  describe "empty input" do
    test "shows no result rather than looking up an empty word", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/lexicon")

      html = view |> form("#lexicon-lookup", %{"word" => "   "}) |> render_submit()

      refute html =~ "Owned facts"
    end
  end
end
