defmodule ChatBotWeb.GettextTest do
  use ExUnit.Case, async: true

  describe "ChatBotWeb.Gettext" do
    test "module is defined" do
      assert Code.ensure_loaded?(ChatBotWeb.Gettext)
    end

    test "uses Gettext.Backend" do
      # Check that the module has Gettext functions
      assert function_exported?(ChatBotWeb.Gettext, :lgettext, 5)
      assert function_exported?(ChatBotWeb.Gettext, :lngettext, 7)
    end

    test "can translate simple strings" do
      # Use the Gettext module directly with the backend
      result = Gettext.gettext(ChatBotWeb.Gettext, "Hello")
      # Returns the string since we don't have translations
      assert result == "Hello"
    end

    test "has __gettext__ function" do
      # All Gettext backends have this function
      assert function_exported?(ChatBotWeb.Gettext, :__gettext__, 1)
    end

    test "otp_app is :chat_bot" do
      assert ChatBotWeb.Gettext.__gettext__(:otp_app) == :chat_bot
    end
  end
end
