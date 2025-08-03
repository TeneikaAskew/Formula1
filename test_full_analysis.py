#!/usr/bin/env python3
"""Run full performance analysis and save output"""

import os
import sys
from pathlib import Path

# Add notebooks/advanced to path
sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "advanced"))

# Change to notebooks/advanced directory
os.chdir(Path(__file__).parent / "notebooks" / "advanced")

try:
    from f1_performance_analysis import test_analyzer
    
    print("Running full F1 Performance Analysis...")
    analyzer, tables = test_analyzer()
    
    # Save the output
    output_file = Path("performance_analysis_output.txt")
    
    # Redirect output to file
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        # Re-run the analysis to capture output
        analyzer.generate_all_tables()
    
    output = f.getvalue()
    
    # Write to file
    output_file.write_text(output)
    
    print(f"\n✓ Analysis completed successfully!")
    print(f"✓ Output saved to: {output_file.absolute()}")
    print(f"✓ File size: {len(output)} characters")
    
    # Show Fantasy overtakes section
    lines = output.split('\n')
    fantasy_start = None
    for i, line in enumerate(lines):
        if "1a. F1 FANTASY OVERTAKE STATISTICS" in line:
            fantasy_start = i
            break
    
    if fantasy_start:
        print("\n" + "="*80)
        print("FANTASY OVERTAKES SECTION PREVIEW:")
        print("="*80)
        # Show 30 lines of the Fantasy section
        for i in range(fantasy_start, min(fantasy_start + 30, len(lines))):
            print(lines[i])
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()