"""Diagnose the import issue"""

import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import requests...")
try:
    import requests
    print("✓ Successfully imported requests")
    print(f"  Location: {requests.__file__}")
except ImportError as e:
    print(f"✗ Failed to import requests: {e}")

print("\nChecking pip list...")
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
if "requests" in result.stdout:
    print("✓ requests is installed according to pip")
else:
    print("✗ requests not found in pip list")

print("\nInstalled packages location:")
result = subprocess.run([sys.executable, "-m", "pip", "show", "requests"], capture_output=True, text=True)
print(result.stdout)