"""
Script to update all F1 notebooks with robust import handling
"""

import json
import os
from pathlib import Path

def create_import_cell():
    """Create the standard import cell with robust path handling"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup imports with robust path handling\n",
            "import sys\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Determine the notebook location and add necessary paths\n",
            "try:\n",
            "    # In Jupyter notebooks, __file__ might not be defined\n",
            "    notebook_dir = Path.cwd()\n",
            "    \n",
            "    # Check if we're in the advanced directory\n",
            "    if 'advanced' in str(notebook_dir):\n",
            "        # We're in the advanced directory\n",
            "        if str(notebook_dir) not in sys.path:\n",
            "            sys.path.insert(0, str(notebook_dir))\n",
            "    else:\n",
            "        # Add the advanced directory to path\n",
            "        workspace_root = notebook_dir\n",
            "        \n",
            "        # Navigate to find the advanced directory\n",
            "        possible_paths = [\n",
            "            notebook_dir / 'notebooks' / 'advanced',  # From workspace root\n",
            "            notebook_dir / 'advanced',  # From notebooks directory\n",
            "            notebook_dir.parent / 'advanced',  # If we're in a sibling directory\n",
            "            notebook_dir.parent / 'notebooks' / 'advanced',  # From other locations\n",
            "        ]\n",
            "        \n",
            "        for path in possible_paths:\n",
            "            if path.exists() and str(path) not in sys.path:\n",
            "                sys.path.insert(0, str(path))\n",
            "                break\n",
            "                \n",
            "except Exception as e:\n",
            "    print(f\"Path setup warning: {e}\")\n",
            "    # Fallback to simple path addition\n",
            "    sys.path.append('.')"
        ]
    }

def update_notebook(notebook_path):
    """Update a notebook with robust import handling"""
    print(f"Updating {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find cells that import f1db_data_loader
    import_cell_indices = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'f1db_data_loader' in source and ('import' in source or 'from' in source):
                import_cell_indices.append(i)
    
    if not import_cell_indices:
        print(f"  No f1db_data_loader imports found in {notebook_path}")
        return False
    
    # Update the first import cell with our robust import logic
    first_import_idx = import_cell_indices[0]
    original_cell = notebook['cells'][first_import_idx]
    
    # Check if it already has our robust import logic
    original_source = ''.join(original_cell.get('source', []))
    if 'possible_paths' in original_source or 'notebook_imports' in original_source:
        print(f"  Already has robust import handling")
        return False
    
    # Create new import cell
    new_import_cell = create_import_cell()
    
    # Extract the actual import statement from the original cell
    import_lines = []
    for line in original_cell.get('source', []):
        if 'f1db_data_loader' in line:
            import_lines.append(line)
    
    # Add the import statement to our new cell
    if import_lines:
        new_import_cell['source'].append("\n")
        new_import_cell['source'].extend(import_lines)
    
    # Replace the original cell
    notebook['cells'][first_import_idx] = new_import_cell
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"  ✓ Updated successfully")
    return True

def main():
    """Update all notebooks with f1db_data_loader imports"""
    notebooks_to_update = [
        'notebooks/advanced/F1_Pipeline_Integration.ipynb',
        'notebooks/advanced/F1_Core_Models.ipynb',
        'notebooks/advanced/F1_Betting_Market_Models.ipynb',
        'notebooks/advanced/F1_Prize_Picks_Optimizer.ipynb',
        'notebooks/advanced/F1_Integrated_Driver_Evaluation.ipynb',
        'notebooks/advanced/F1_Feature_Store.ipynb',
        'notebooks/advanced/F1_Backtesting_Framework.ipynb',
        'notebooks/advanced/F1_Explainability_Engine.ipynb',
        'notebooks/advanced/F1_Constructor_Driver_Evaluation.ipynb',
        'notebooks/advanced/F1DB_Data_Tutorial.ipynb',
    ]
    
    workspace_root = Path(__file__).parent
    updated_count = 0
    
    for notebook_rel_path in notebooks_to_update:
        notebook_path = workspace_root / notebook_rel_path
        if notebook_path.exists():
            if update_notebook(notebook_path):
                updated_count += 1
        else:
            print(f"⚠ Notebook not found: {notebook_path}")
    
    print(f"\n✓ Updated {updated_count} notebooks")

if __name__ == "__main__":
    main()