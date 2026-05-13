#!/usr/bin/env python3
"""
Quick local test runner — runs all pytest tests.
"""
import subprocess
import sys

result = subprocess.run(
    ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=False,
)
sys.exit(result.returncode)
