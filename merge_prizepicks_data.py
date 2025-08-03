#!/usr/bin/env python3
"""
Merge new PrizePicks data with existing data to avoid duplicates
"""

import os
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_prizepicks_data():
    """Merge all PrizePicks CSV files into a single deduplicated file"""
    
    data_dir = "data/prizepicks"
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist")
        return
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'prizepicks_lineups_merged.csv']
    
    if not csv_files:
        logger.warning("No CSV files found to merge")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to merge")
    
    # Read all CSV files
    all_dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} rows from {csv_file}")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {str(e)}")
    
    if not all_dfs:
        logger.error("No data frames to merge")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined data has {len(combined_df)} total rows")
    
    # Remove duplicates based on lineup_id
    if 'lineup_id' in combined_df.columns:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['lineup_id'], keep='last')
        after_dedup = len(combined_df)
        logger.info(f"Removed {before_dedup - after_dedup} duplicate lineups")
    
    # Sort by date (if available)
    if 'date' in combined_df.columns:
        combined_df = combined_df.sort_values('date', ascending=False)
    
    # Save merged file
    output_path = os.path.join(data_dir, 'prizepicks_lineups_merged.csv')
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(combined_df)} unique lineups to {output_path}")
    
    # Create summary statistics
    if 'result' in combined_df.columns:
        wins = len(combined_df[combined_df['result'] == 'Win'])
        losses = len(combined_df[combined_df['result'] == 'Loss'])
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        
        logger.info("\nSummary Statistics:")
        logger.info(f"Total Lineups: {len(combined_df)}")
        logger.info(f"Wins: {wins}")
        logger.info(f"Losses: {losses}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        
        if 'entry_fee' in combined_df.columns and 'payout' in combined_df.columns:
            # Convert currency strings to float
            combined_df['entry_fee_float'] = combined_df['entry_fee'].str.replace('$', '').astype(float)
            combined_df['payout_float'] = combined_df['payout'].str.replace('$', '').astype(float)
            
            total_invested = combined_df['entry_fee_float'].sum()
            total_payout = combined_df[combined_df['result'] == 'Win']['payout_float'].sum()
            net_profit = total_payout - total_invested
            roi = (net_profit / total_invested * 100) if total_invested > 0 else 0
            
            logger.info(f"Total Invested: ${total_invested:.2f}")
            logger.info(f"Total Payout: ${total_payout:.2f}")
            logger.info(f"Net Profit: ${net_profit:.2f}")
            logger.info(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    merge_prizepicks_data()