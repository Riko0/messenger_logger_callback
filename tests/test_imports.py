"""Verify that both interfaces import correctly."""

import unittest
import sys
from unittest import mock


class TestImports(unittest.TestCase):
    def test_standalone_import(self):
        from messenger_logger import MessengerLogger
        self.assertTrue(callable(MessengerLogger))

    def test_callback_import(self):
        from messenger_logger.callback import MessengerLoggerCallback
        self.assertTrue(callable(MessengerLoggerCallback))

    def test_engine_import(self):
        from messenger_logger.engine import LoggerEngine
        self.assertTrue(callable(LoggerEngine))

    def test_top_level_reexports(self):
        import messenger_logger
        self.assertTrue(hasattr(messenger_logger, "MessengerLogger"))
        self.assertTrue(hasattr(messenger_logger, "MessengerLoggerCallback"))

    def test_import_without_transformers(self):
        """MessengerLogger must be importable even if transformers is missing."""
        mods_to_clear = [
            k for k in sys.modules if k.startswith("messenger_logger")
        ]
        saved = {}
        for k in mods_to_clear:
            saved[k] = sys.modules.pop(k)

        try:
            with mock.patch.dict(sys.modules, {"transformers": None}):
                from messenger_logger import MessengerLogger  # noqa: F811
                self.assertTrue(callable(MessengerLogger))
        finally:
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
