"""Hard-coded local settings, and the chat-id discovery helper."""

import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

import config  # noqa: E402


class LocalConfigCase(unittest.TestCase):
    """Writes a throwaway local_config.py onto sys.path for each test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        sys.path.insert(0, self.tmp.name)
        self._env = dict(os.environ)
        sys.modules.pop("local_config", None)

    def tearDown(self):
        sys.path.remove(self.tmp.name)
        sys.modules.pop("local_config", None)
        os.environ.clear()
        os.environ.update(self._env)
        self.tmp.cleanup()

    def write(self, body):
        with open(os.path.join(self.tmp.name, "local_config.py"), "w",
                  encoding="utf-8") as fh:
            fh.write(body)


class TestLocalConfig(LocalConfigCase):
    def test_values_are_applied_when_the_environment_is_empty(self):
        self.write('TELEGRAM_BOT_TOKEN = "1:AAtoken"\nTELEGRAM_CHAT_ID = 12345\n')
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)

        config._apply_local_config()

        self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "1:AAtoken")
        self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "12345", "numbers are stringified")

    def test_a_real_environment_variable_always_wins(self):
        self.write('TELEGRAM_CHAT_ID = "from_file"\n')
        os.environ["TELEGRAM_CHAT_ID"] = "from_environment"

        config._apply_local_config()

        self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "from_environment")

    def test_lowercase_and_private_names_are_ignored(self):
        self.write('helper = "x"\n_SECRET = "y"\nSYMBOLS = "BTC/USDT"\n')
        os.environ.pop("SYMBOLS", None)

        config._apply_local_config()

        self.assertEqual(os.environ.get("SYMBOLS"), "BTC/USDT")
        self.assertNotIn("helper", os.environ)
        self.assertNotIn("_SECRET", os.environ)

    def test_callables_and_none_are_skipped(self):
        self.write('def FUNC():\n    return 1\nEMPTY = None\nOK = "yes"\n')
        os.environ.pop("OK", None)

        config._apply_local_config()

        self.assertEqual(os.environ.get("OK"), "yes")
        self.assertNotIn("FUNC", os.environ)
        self.assertNotIn("EMPTY", os.environ)

    def test_a_broken_file_is_ignored_instead_of_crashing(self):
        self.write("this is not valid python !!!\n")
        config._apply_local_config()          # must not raise

    def test_absent_file_is_fine(self):
        config._apply_local_config()          # nothing written; must not raise

    def test_settings_pick_the_values_up(self):
        self.write('TELEGRAM_BOT_TOKEN = "9:AAbc"\nTELEGRAM_CHAT_ID = "42"\n'
                   'MIN_CONFIDENCE = 3\n')
        for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "MIN_CONFIDENCE"):
            os.environ.pop(name, None)

        config._apply_local_config()
        settings = config.load_settings()

        self.assertEqual(settings.telegram_bot_token, "9:AAbc")
        self.assertEqual(settings.telegram_chat_id, "42")
        self.assertEqual(settings.min_confidence, 3)
        settings.require_telegram()           # must not raise


class TestFindChatId(unittest.TestCase):
    def test_without_a_token_it_says_so_and_exits_two(self):
        from dataclasses import replace

        from notifier import discover_chats

        code = discover_chats(replace(config.Settings(), telegram_bot_token=""))
        self.assertEqual(code, 2)

    def test_it_lists_the_chats_telegram_reports(self):
        from dataclasses import replace

        import notifier

        async def fake(token):
            return [
                {"id": 1234567890, "type": "private", "name": "Sam"},
                {"id": -1009876543210, "type": "supergroup", "name": "Signals Group"},
            ]

        printed = []
        with mock.patch.object(notifier, "_discover_chats_async", fake), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = notifier.discover_chats(
                replace(config.Settings(), telegram_bot_token="1:AA"))

        self.assertEqual(code, 0)
        joined = "\n".join(printed)
        self.assertIn("TELEGRAM_CHAT_ID=1234567890", joined)
        self.assertIn("TELEGRAM_CHAT_ID=-1009876543210", joined)
        self.assertIn("Signals Group", joined)

    def test_no_chats_yet_explains_the_next_step(self):
        from dataclasses import replace

        import notifier

        async def none(token):
            return []

        printed = []
        with mock.patch.object(notifier, "_discover_chats_async", none), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = notifier.discover_chats(
                replace(config.Settings(), telegram_bot_token="1:AA"))

        self.assertEqual(code, 1)
        self.assertIn("Send your bot a message", "\n".join(printed))


if __name__ == "__main__":
    unittest.main()
