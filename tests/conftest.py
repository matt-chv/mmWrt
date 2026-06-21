"""
Module-level code in test_assets.py runs when it's imported, not when the test runs.
to avoid importing mmWrt in every test file defining it here"""

import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))