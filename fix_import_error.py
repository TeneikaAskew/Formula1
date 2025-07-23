"""
Quick fix for f1db_data_loader import errors in Jupyter notebooks
"""

import sys
import os

def setup_f1_imports():
    """Add necessary paths for F1 project imports"""
    # Get the workspace root directory
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add paths for different notebook locations
    paths_to_add = [
        workspace_root,  # Root directory
        os.path.join(workspace_root, 'notebooks', 'advanced'),  # Advanced notebooks
        os.path.join(workspace_root, 'notebooks'),  # Notebooks directory
    ]
    
    for path in paths_to_add:
        if path not in sys.path and os.path.exists(path):
            sys.path.append(path)
            print(f"Added to Python path: {path}")
    
    print("\nYou should now be able to import f1db_data_loader")
    print("Example usage:")
    print("  from f1db_data_loader import load_f1db_data")
    print("  f1_data = load_f1db_data()")

if __name__ == "__main__":
    setup_f1_imports()