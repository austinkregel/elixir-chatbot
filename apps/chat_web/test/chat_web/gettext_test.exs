defmodule ChatWeb.GettextTest do
  use ExUnit.Case, async: true

  describe "ChatWeb.Gettext" do
    test "module is defined" do
      assert Code.ensure_loaded?(ChatWeb.Gettext)
    end

    test "uses Gettext.Backend" do
      # Check that the module has Gettext functions
      assert function_exported?(ChatWeb.Gettext, :lgettext, 5)
      assert function_exported?(ChatWeb.Gettext, :lngettext, 7)
    end

    test "can translate simple strings" do
      # Use the Gettext module directly with the backend
      result = Gettext.gettext(ChatWeb.Gettext, "Hello")
      # Returns the string since we don't have translations
      assert result == "Hello"
    end

    test "has __gettext__ function" do
      # All Gettext backends have this function
      assert function_exported?(ChatWeb.Gettext, :__gettext__, 1)
    end

    test "otp_app is :chat_web" do
      assert ChatWeb.Gettext.__gettext__(:otp_app) == :chat_web
    end
  end
end
