#!/usr/bin/env python3
"""
Test script for DHL pit stop data integration with F1 performance analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def test_dhl_integration():
    """Test the DHL pit stop integration"""
    print("="*60)
    print("TESTING DHL PIT STOP DATA INTEGRATION")
    print("="*60)
    
    try:
        # Load F1 database
        print("Loading F1 database...")
        data = load_f1db_data()
        print(f"✓ Loaded {len(data)} data tables")
        
        # Create analyzer
        print("\nCreating performance analyzer...")
        analyzer = F1PerformanceAnalyzer(data)
        
        # Check if DHL data was loaded
        if analyzer.dhl_data.empty:
            print("⚠️  No DHL data found - this is expected if no CSV files exist yet")
            print("   Run the DHL scraper first: python dhl_current_season_scraper.py")
            return False
        else:
            print(f"✓ DHL data loaded: {len(analyzer.dhl_data)} pit stop records")
            print(f"  Races: {analyzer.dhl_data['race'].nunique() if 'race' in analyzer.dhl_data.columns else 'N/A'}")
            print(f"  Years: {analyzer.dhl_data['year'].unique().tolist() if 'year' in analyzer.dhl_data.columns else 'N/A'}")
        
        # Test DHL analysis method
        print("\nTesting DHL pit stop analysis...")
        dhl_analysis = analyzer.analyze_dhl_pit_stops()
        
        if dhl_analysis.empty:
            print("⚠️  DHL analysis returned no results")
            print("   This might be due to driver name matching issues")
            return False
        else:
            print(f"✓ DHL analysis successful: {len(dhl_analysis)} drivers analyzed")
            print(f"  Columns: {dhl_analysis.columns.tolist()}")
            
            # Show sample results
            if len(dhl_analysis) > 0:
                print("\nSample DHL Results:")
                print("-" * 40)
                for i, (driver_id, row) in enumerate(dhl_analysis.head(3).iterrows()):
                    print(f"Driver {driver_id}: Avg={row['avg_time']:.3f}s, Best={row['best_time']:.3f}s, Stops={row['total_stops']}")
        
        # Test full performance analysis (just the first few sections)
        print("\nTesting integration with full performance analysis...")
        print("(This will show just the DHL section)")
        print("\n" + "-"*60)
        
        # Run just the DHL section
        dhl_stops = analyzer.analyze_dhl_pit_stops()
        if not dhl_stops.empty:
            drivers = analyzer.data.get('drivers', pd.DataFrame())
            if not drivers.empty:
                driver_names = drivers.set_index('id')['surname'].to_dict()
                dhl_stops['driver_name'] = dhl_stops.index.map(driver_names)
            
            display_cols = ['driver_name', 'avg_time', 'median_time', 'best_time', 'best_time_lap', 'total_stops']
            display_cols = [col for col in display_cols if col in dhl_stops.columns]
            
            print("3b. DHL OFFICIAL PIT STOP TIMES BY DRIVER (seconds)")
            print("-" * 60)
            dhl_stops_sorted = dhl_stops.sort_values('avg_time', ascending=True)
            print(dhl_stops_sorted[display_cols].head(10).to_string())
        
        print("\n" + "="*60)
        print("✅ DHL INTEGRATION TEST PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import pandas as pd
    success = test_dhl_integration()
    if success:
        print("\n🎯 Ready to use DHL data in the F1 pipeline!")
        print("   Run: python notebooks/advanced/run_f1_pipeline.py")
    else:
        print("\n⚠️  Please check the issues above before running the pipeline")