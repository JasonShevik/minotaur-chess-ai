"""
Puts the repo's root folder on sys.path, so that a test finds the code it tests wherever it is started from:

    python tests/test_x.py          from the repo's root
    python test_x.py                from inside tests/

Every test imports this before its project imports. ROOT is the root folder, for the few tests that look
for a file there.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
