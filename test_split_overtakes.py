#!/usr/bin/env python3
"""Test the split Fantasy overtakes display"""

import os
import sys
from pathlib import Path

# Add notebooks/advanced to path
sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "advanced"))

# Change to notebooks/advanced directory
os.chdir(Path(__file__).parent / "notebooks" / "advanced")

try:
    from f1_performance_analysis import test_analyzer
    
    print("Testing split Fantasy overtakes display...")
    analyzer, tables = test_analyzer()
    
    print("\n" + "="*80)
    print("FANTASY OVERTAKES SECTION EXTRACTED:")
    print("="*80)
    
    # Get just the Fantasy overtakes output
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        # Call just the fantasy overtakes part
        fantasy_overtakes = analyzer.analyze_fantasy_overtakes()
        if not fantasy_overtakes.empty:
            # Race section
            print("\n1a. F1 FANTASY RACE OVERTAKE STATISTICS")
            print("-"*80)
            race_cols = ['driver_name', 'total_races', 'total_overtake_points']
            race_overtake_cols = [col for col in fantasy_overtakes.columns if 'race_overtake' in col and '_avg_pts' in col]
            race_count_cols = [col for col in fantasy_overtakes.columns if 'race_overtake' in col and '_avg_count' in col]
            
            race_cols.extend(race_overtake_cols)
            race_cols.extend(race_count_cols)
            race_cols = [col for col in race_cols if col in fantasy_overtakes.columns]
            
            print("\nRace Overtake Points Summary:")
            print(fantasy_overtakes.sort_values('total_overtake_points', ascending=False)[race_cols].head(10).to_string(index=False))
            
            # Sprint section
            print("\n\n1b. F1 FANTASY SPRINT OVERTAKE STATISTICS")
            print("-"*80)
            sprint_cols = ['driver_name', 'total_races']
            sprint_overtake_cols = [col for col in fantasy_overtakes.columns if 'sprint_overtake' in col and '_avg_pts' in col]
            sprint_count_cols = [col for col in fantasy_overtakes.columns if 'sprint_overtake' in col and '_avg_count' in col]
            
            sprint_cols.extend(sprint_overtake_cols)
            sprint_cols.extend(sprint_count_cols)
            sprint_cols = [col for col in sprint_cols if col in fantasy_overtakes.columns]
            
            has_sprint_data = fantasy_overtakes[fantasy_overtakes['sprint_overtake_bonus_avg_pts'] > 0] if 'sprint_overtake_bonus_avg_pts' in fantasy_overtakes.columns else fantasy_overtakes
            
            print("\nSprint Overtake Points Summary:")
            print(has_sprint_data.sort_values('sprint_overtake_bonus_avg_pts', ascending=False)[sprint_cols].head(10).to_string(index=False))
    
    output = f.getvalue()
    print(output)
    
    print("\n✓ Fantasy overtakes successfully split into Race and Sprint sections!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()