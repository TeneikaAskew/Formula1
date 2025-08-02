#!/usr/bin/env python3
"""
Test script for the enhanced F1 pipeline
Tests configuration loading, data availability, and pipeline components
"""

import sys
import os
import yaml
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test if pipeline configuration loads correctly"""
    print("🔧 Testing configuration loading...")
    
    config_path = Path("pipeline_config_enhanced.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['pipeline', 'data', 'performance_analysis', 'predictions_v3', 'predictions_v4']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print(f"✅ Configuration loaded: {config['pipeline']['name']} v{config['pipeline']['version']}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_data_availability():
    """Test if F1DB data is available"""
    print("📊 Testing data availability...")
    
    try:
        from f1db_data_loader import F1DBDataLoader
        loader = F1DBDataLoader()
        
        # Test core datasets loading
        start_time = time.time()
        data = loader.get_core_datasets()
        load_time = time.time() - start_time
        
        print(f"✅ F1DB data loaded in {load_time:.2f}s")
        
        # Check key datasets
        expected_datasets = ['races', 'drivers', 'results', 'constructors']
        for dataset in expected_datasets:
            if dataset in data and len(data[dataset]) > 0:
                print(f"  📋 {dataset}: {len(data[dataset])} records")
            else:
                print(f"  ❌ Missing or empty dataset: {dataset}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("📦 Testing dependencies...")
    
    required_modules = [
        'f1db_data_loader',
        'f1_performance_analysis', 
        'f1_predictions_enhanced_v3',
        'f1_predictions_enhanced_v3_weather',
        'f1_predictions_v4_production'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing {len(missing_modules)} required modules")
        return False
    
    print("✅ All dependencies available")
    return True

def test_output_directories():
    """Test if output directories can be created"""
    print("📁 Testing output directories...")
    
    import tempfile
    directories = [
        "pipeline_outputs",
        str(Path(tempfile.gettempdir()) / "f1db_cache")
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            if Path(directory).exists():
                print(f"  ✅ {directory}")
            else:
                print(f"  ❌ {directory}: Cannot create")
                return False
        except Exception as e:
            print(f"  ❌ {directory}: {e}")
            return False
    
    print("✅ Output directories ready")
    return True

def test_pipeline_orchestrator():
    """Test if pipeline orchestrator can be instantiated"""
    print("🚀 Testing pipeline orchestrator...")
    
    try:
        from run_enhanced_pipeline import EnhancedF1Pipeline
        
        # Test instantiation
        pipeline = EnhancedF1Pipeline()
        print(f"  ✅ Pipeline instantiated: {pipeline.config['pipeline']['name']}")
        
        # Test logging setup
        if pipeline.logger:
            print("  ✅ Logging configured")
        else:
            print("  ❌ Logging not configured")
            return False
        
        # Test configuration loading
        if pipeline.config:
            print("  ✅ Configuration loaded")
        else:
            print("  ❌ Configuration not loaded")
            return False
        
        print("✅ Pipeline orchestrator ready")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline orchestrator failed: {e}")
        return False

def run_quick_pipeline_test():
    """Run a quick test of the data loading component"""
    print("⚡ Running quick pipeline test...")
    
    try:
        from run_enhanced_pipeline import EnhancedF1Pipeline
        
        pipeline = EnhancedF1Pipeline()
        
        # Test data loading
        start_time = time.time()
        pipeline.load_shared_data()
        load_time = time.time() - start_time
        
        if pipeline.data:
            print(f"  ✅ Shared data loaded in {load_time:.2f}s")
            
            # Check data content
            if 'races' in pipeline.data and len(pipeline.data['races']) > 0:
                print(f"  📊 Data contains {len(pipeline.data['races'])} races")
            
            # Test caching
            if pipeline.data_cache_file and pipeline.data_cache_file.exists():
                cache_size = pipeline.data_cache_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  💾 Data cached: {cache_size:.1f}MB")
            
            print("✅ Quick pipeline test passed")
            return True
        else:
            print("  ❌ No data loaded")
            return False
        
    except Exception as e:
        print(f"❌ Quick pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED F1 PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config_loading),
        ("Dependencies", test_dependencies),
        ("Output Directories", test_output_directories),
        ("Data Availability", test_data_availability),
        ("Pipeline Orchestrator", test_pipeline_orchestrator),
        ("Quick Pipeline Test", run_quick_pipeline_test),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Pipeline is ready to run.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix issues before running pipeline.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)