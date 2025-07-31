"""
Test the new overtakes by track and year analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def test_overtakes_analysis():
    """Test the overtakes analysis functionality"""
    print("Loading F1 data...")
    data = load_f1db_data()
    
    print("Creating analyzer...")
    analyzer = F1PerformanceAnalyzer(data)
    
    print("\nTesting track-specific overtakes by year analysis...")
    track_analysis = analyzer.analyze_overtakes_by_track_year()
    
    if isinstance(track_analysis, dict):
        print(f"\n✓ Analysis returned for circuit: {track_analysis.get('circuit_name', 'Unknown')}")
        
        if 'year_by_year' in track_analysis:
            year_data = track_analysis['year_by_year']
            print(f"✓ Year-by-year data: {len(year_data)} entries")
            if not year_data.empty:
                print("\nSample year-by-year data:")
                print(year_data[['driver_name', 'year', 'avg_overtakes', 'avg_start_pos', 'avg_finish_pos']].head(10))
        
        if 'overall_stats' in track_analysis:
            overall = track_analysis['overall_stats']
            print(f"\n✓ Overall stats: {len(overall)} drivers")
            if not overall.empty:
                print("\nTop 5 drivers by average overtakes at this track:")
                top_5 = overall.nlargest(5, 'career_avg_overtakes')
                print(top_5[['driver_name', 'races_at_track', 'career_avg_overtakes', 'career_avg_start', 'career_avg_finish']])
    else:
        print("✗ No track analysis data returned")
    
    print("\n" + "="*60)
    print("Running full performance analysis tables...")
    print("="*60)
    analyzer.generate_all_tables()

if __name__ == "__main__":
    test_overtakes_analysis()