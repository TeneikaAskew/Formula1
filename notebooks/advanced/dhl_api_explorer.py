#!/usr/bin/env python3
"""
DHL API Explorer - Examine the structure of DHL API responses
"""

import requests
import json
from pprint import pprint

def explore_api_endpoint(endpoint_id: str, description: str):
    """Explore a single API endpoint"""
    url = f"https://inmotion.dhl/api/f1-award-element-data/{endpoint_id}"
    
    print(f"\n{'='*80}")
    print(f"Exploring: {description}")
    print(f"URL: {url}")
    print(f"{'='*80}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Show the structure
        print(f"\nResponse type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            
            # Explore each key
            for key in data.keys():
                value = data[key]
                print(f"\nKey '{key}':")
                print(f"  Type: {type(value)}")
                
                if isinstance(value, list):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First item type: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"  First item keys: {list(value[0].keys())}")
                            print(f"  First item sample:")
                            pprint(value[0], indent=4, depth=2)
                elif isinstance(value, dict):
                    print(f"  Keys: {list(value.keys())}")
                    # Show a sample
                    sample_key = list(value.keys())[0] if value else None
                    if sample_key:
                        print(f"  Sample value for key '{sample_key}':")
                        pprint(value[sample_key], indent=4, depth=2)
                else:
                    print(f"  Value: {value}")
        
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
                    print(f"First item:")
                    pprint(data[0], indent=2)
        
        # Save raw response for analysis
        filename = f"dhl_api_response_{endpoint_id}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nRaw response saved to: {filename}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Explore all DHL API endpoints"""
    
    endpoints = [
        ('6273', '2024 Drivers and Times'),
        ('6276', '2024 Races'),
        ('6282', '2023 Drivers and Times'),
        ('6284', '2023 Races'),
        ('6365', '2022 Drivers/Times/Races')
    ]
    
    print("DHL API Structure Explorer")
    print("=" * 80)
    
    for endpoint_id, description in endpoints:
        explore_api_endpoint(endpoint_id, description)


if __name__ == "__main__":
    main()