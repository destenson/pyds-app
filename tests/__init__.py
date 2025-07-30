"""
PyDS test suite for comprehensive system validation.

This test suite provides unit, integration, and performance tests for the
DeepStream inference system with mock GStreamer/DeepStream components.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))