"""Run equation and batch integration tests with mocked neural encoders (requires Pillow)."""
from pathlib import Path
import unittest

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
