#!/usr/bin/env python3
"""Test v4 with automated inputs"""

import sys
import io

# Provide default input
sys.stdin = io.StringIO('1\n')

# Run the simple runner
from run_v4_simple import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()