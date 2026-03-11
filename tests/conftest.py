"""
Pytest configuration file.
Adds the workspace root to Python path for imports.
"""

import sys
import os

# Add workspace root to Python path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
