#!/usr/bin/env python3
"""
Parse PrizePicks wager data from API response
"""

import json
import pandas as pd
from datetime import datetime
import os
import sys

def parse_wager_data(file_path):
    """Parse wager data from PrizePicks API response"""
    
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prediction_rows = []  # Changed from wagers to prediction_rows
    predictions = []
    projections = []
    players = []
    scores = []
    teams = []
    
    # Extract included data (predictions, projections, players, scores, teams)
    if 'included' in data:
        for item in data['included']:
            if item['type'] == 'prediction':
                predictions.append(item)
            elif item['type'] == 'projection':
                projections.append(item)
            elif item['type'] == 'new_player':
                players.append(item)
            elif item['type'] == 'score':
                scores.append(item)
            elif item['type'] == 'team':
                teams.append(item)
    
    # Create lookup dictionaries
    prediction_map = {p['id']: p for p in predictions}
    projection_map = {p['id']: p for p in projections}
    player_map = {p['id']: p for p in players}
    score_map = {s['id']: s for s in scores}
    team_map = {t['id']: t for t in teams}
    
    # Process wagers - create one row per prediction
    for wager in data['data']:
        if wager['type'] != 'new_wager':
            continue
            
        w = wager['attributes']
        wager_id = wager['id']
        
        # Calculate wager-level data
        is_wager_win = w['result'] in ['won', 'partial_win']
        actual_payout = w['amount_won_cents'] / 100.0
        bet_amount = w['amount_bet_cents'] / 100.0
        potential_payout = w['amount_to_win_cents'] / 100.0
        profit_loss = actual_payout - bet_amount
        pick_type = f"{w['parlay_count']}-Pick"
        play_type = 'Power Play' if not w.get('pick_protection', True) else 'Flex Play'
        
        # Process each prediction in this wager
        if 'predictions' in wager['relationships'] and wager['relationships']['predictions']['data']:
            for pred_ref in wager['relationships']['predictions']['data']:
                pred_id = pred_ref['id']
                if pred_id not in prediction_map:
                    continue
                    
                pred = prediction_map[pred_id]
                pred_attrs = pred['attributes']
                
                # Initialize prediction row with wager data
                prediction_row = {
                    'wager_id': wager_id,
                    'prediction_id': pred_id,
                    'date': datetime.fromisoformat(w['created_at'].replace('-04:00', '-04:00')).strftime('%B %d, %Y'),
                    'sport': w.get('sport', 'Unknown'),
                    'pick_type': pick_type,
                    'play_type': play_type,
                    'entry_fee': f"${bet_amount:.2f}",
                    'potential_payout': f"${potential_payout:.2f}",
                    'actual_payout': actual_payout,
                    'wager_result': 'Win' if is_wager_win else 'Loss',
                    'wager_profit_loss': profit_loss,
                    'created_at': w['created_at'],
                    'updated_at': w['updated_at']
                }
                
                # Add prediction-level data
                prediction_row.update({
                    'line_score': pred_attrs.get('line_score', ''),
                    'wager_type': pred_attrs.get('wager_type', ''),  # over/under
                    'odds_type': pred_attrs.get('odds_type', ''),
                    'is_promo': pred_attrs.get('is_promo', False)
                })
                
                # Get projection details
                if 'projection' in pred['relationships'] and pred['relationships']['projection']['data']:
                    proj_id = pred['relationships']['projection']['data']['id']
                    if proj_id in projection_map:
                        proj = projection_map[proj_id]
                        proj_attrs = proj['attributes']
                        
                        prediction_row.update({
                            'projection_id': proj_id,
                            'stat_type': proj_attrs.get('stat_type', ''),
                            'description': proj_attrs.get('description', ''),
                            'start_time': proj_attrs.get('start_time', ''),
                            'board_time': proj_attrs.get('board_time', ''),
                            'is_flex': proj_attrs.get('is_flex', False)
                        })
                
                # Get player details
                player_name = 'Unknown'
                if 'new_player' in pred['relationships'] and pred['relationships']['new_player']['data']:
                    player_id = pred['relationships']['new_player']['data']['id']
                    if player_id in player_map:
                        player_attrs = player_map[player_id]['attributes']
                        player_name = player_attrs.get('display_name', 'Unknown')
                        prediction_row.update({
                            'player_id': player_id,
                            'player_name': player_name,
                            'position': player_attrs.get('position', ''),
                            'team_id': player_attrs.get('team_id', '')
                        })
                
                # Get actual score/result
                actual_score = None
                is_final = False
                if 'score' in pred['relationships'] and pred['relationships']['score']['data']:
                    score_id = pred['relationships']['score']['data']['id']
                    if score_id in score_map:
                        score_attrs = score_map[score_id]['attributes']
                        actual_score = score_attrs.get('score', None)
                        is_final = score_attrs.get('is_final', False)
                        prediction_row.update({
                            'score_id': score_id,
                            'actual_score': actual_score,
                            'is_final': is_final,
                            'is_off_the_board': score_attrs.get('is_off_the_board', False)
                        })
                
                # Calculate prediction result
                line_score = pred_attrs.get('line_score', 0)
                wager_type = pred_attrs.get('wager_type', '').lower()
                
                if actual_score is not None and line_score is not None:
                    if wager_type == 'over':
                        prediction_result = 'Won' if actual_score > line_score else 'Lost'
                        if actual_score == line_score:
                            prediction_result = 'Push'
                    elif wager_type == 'under':
                        prediction_result = 'Won' if actual_score < line_score else 'Lost'
                        if actual_score == line_score:
                            prediction_result = 'Push'
                    else:
                        prediction_result = 'Unknown'
                else:
                    prediction_result = 'Unknown'
                
                prediction_row['prediction_result'] = prediction_result
                
                # Add to results
                prediction_rows.append(prediction_row)
    
    return pd.DataFrame(prediction_rows)

def generate_summary(df):
    """Generate summary statistics"""
    
    # Clean up numeric columns
    df['entry_fee_amount'] = df['entry_fee'].str.replace('$', '').astype(float)
    df['potential_payout_amount'] = df['potential_payout'].str.replace('$', '').astype(float)
    
    # Summary stats at wager level (aggregate predictions per wager)
    wager_summary = df.groupby('wager_id').agg({
        'entry_fee_amount': 'first',
        'actual_payout': 'first', 
        'wager_profit_loss': 'first',
        'wager_result': 'first',
        'sport': 'first',
        'pick_type': 'first'
    }).reset_index()
    
    total_invested = wager_summary['entry_fee_amount'].sum()
    total_payout = wager_summary['actual_payout'].sum()
    net_profit = wager_summary['wager_profit_loss'].sum()
    
    summary = {
        'total_wagers': len(wager_summary),
        'total_predictions': len(df),
        'wager_wins': len(wager_summary[wager_summary['wager_result'] == 'Win']),
        'wager_losses': len(wager_summary[wager_summary['wager_result'] == 'Loss']),
        'wager_win_rate': (len(wager_summary[wager_summary['wager_result'] == 'Win']) / len(wager_summary) * 100) if len(wager_summary) > 0 else 0,
        'prediction_wins': len(df[df['prediction_result'] == 'Won']),
        'prediction_losses': len(df[df['prediction_result'] == 'Lost']),
        'prediction_win_rate': (len(df[df['prediction_result'] == 'Won']) / len(df) * 100) if len(df) > 0 else 0,
        'total_invested': total_invested,
        'total_payout': total_payout,
        'net_profit_loss': net_profit,
        'roi': (net_profit / total_invested * 100) if total_invested > 0 else 0
    }
    
    # By sport (wager level)
    by_sport = wager_summary.groupby('sport').agg({
        'wager_id': 'count',
        'wager_result': lambda x: (x == 'Win').sum(),
        'wager_profit_loss': 'sum',
        'entry_fee_amount': 'sum'
    }).rename(columns={
        'wager_id': 'total_wagers',
        'wager_result': 'wins',
        'wager_profit_loss': 'net_profit',
        'entry_fee_amount': 'invested'
    })
    by_sport['win_rate'] = (by_sport['wins'] / by_sport['total_wagers'] * 100).round(1)
    by_sport['roi'] = (by_sport['net_profit'] / by_sport['invested'] * 100).round(1)
    
    # By pick type (wager level)
    by_pick_type = wager_summary.groupby('pick_type').agg({
        'wager_id': 'count',
        'wager_result': lambda x: (x == 'Win').sum(),
        'wager_profit_loss': 'sum'
    }).rename(columns={
        'wager_id': 'total',
        'wager_result': 'wins',
        'wager_profit_loss': 'net_profit'
    })
    by_pick_type['win_rate'] = (by_pick_type['wins'] / by_pick_type['total'] * 100).round(1)
    
    return summary, by_sport, by_pick_type

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python parse_prizepicks_wagers.py <lineup.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Parse data
        df = parse_wager_data(file_path)
        print(f"\n✓ Parsed {len(df)} predictions from {df['wager_id'].nunique()} wagers")
        
        # Sort by date
        df['created_datetime'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_datetime', ascending=False)
        
        # Generate summary
        summary, by_sport, by_pick_type = generate_summary(df)
        
        # Print summary
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)
        print(f"Total Wagers: {summary['total_wagers']}")
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"\nWager Performance:")
        print(f"  Wins: {summary['wager_wins']} ({summary['wager_win_rate']:.1f}%)")
        print(f"  Losses: {summary['wager_losses']}")
        print(f"\nPrediction Performance:")
        print(f"  Wins: {summary['prediction_wins']} ({summary['prediction_win_rate']:.1f}%)")
        print(f"  Losses: {summary['prediction_losses']}")
        print(f"\nFinancial Summary:")
        print(f"  Total Invested: ${summary['total_invested']:,.2f}")
        print(f"  Total Payout: ${summary['total_payout']:,.2f}")
        print(f"  Net Profit/Loss: ${summary['net_profit_loss']:,.2f}")
        print(f"  ROI: {summary['roi']:.1f}%")
        
        print("\n" + "="*50)
        print("BY SPORT")
        print("="*50)
        print(by_sport.to_string())
        
        print("\n" + "="*50)
        print("BY PICK TYPE")
        print("="*50)
        print(by_pick_type.to_string())
        
        # Save to CSV
        output_dir = "data/prizepicks"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save predictions/lineups CSV (one row per prediction)
        predictions_csv_path = os.path.join(output_dir, f"prizepicks_lineups_{timestamp}.csv")
        
        # Select key columns for predictions CSV
        prediction_columns = ['wager_id', 'prediction_id', 'date', 'sport', 'pick_type', 'play_type', 
                            'player_name', 'position', 'team_id', 'stat_type', 'description',
                            'line_score', 'wager_type', 'actual_score', 'prediction_result', 
                            'is_final', 'start_time', 'created_at']
        
        # Only include columns that exist
        available_prediction_cols = [col for col in prediction_columns if col in df.columns]
        df[available_prediction_cols].to_csv(predictions_csv_path, index=False)
        
        print(f"\n✓ Saved lineups to: {predictions_csv_path}")
        
        # Also save latest lineups
        latest_predictions_path = os.path.join(output_dir, "prizepicks_lineups_latest.csv")
        df[available_prediction_cols].to_csv(latest_predictions_path, index=False)
        
        # 2. Save wagers CSV (one row per wager)
        # Aggregate to wager level
        wager_df = df.groupby('wager_id').agg({
            'date': 'first',
            'sport': 'first',
            'pick_type': 'first',
            'play_type': 'first',
            'entry_fee': 'first',
            'potential_payout': 'first',
            'actual_payout': 'first',
            'wager_result': 'first',
            'wager_profit_loss': 'first',
            'created_at': 'first',
            'updated_at': 'first',
            'prediction_id': 'count',  # count of predictions
            'player_name': lambda x: ', '.join(x.dropna().unique()),  # list of players
            'prediction_result': lambda x: f"{(x == 'Won').sum()}/{len(x)}"  # wins/total
        }).reset_index()
        
        # Rename columns
        wager_df = wager_df.rename(columns={
            'prediction_id': 'num_predictions',
            'player_name': 'players',
            'prediction_result': 'predictions_won'
        })
        
        wagers_csv_path = os.path.join(output_dir, f"prizepicks_wagers_{timestamp}.csv")
        wager_df.to_csv(wagers_csv_path, index=False)
        
        print(f"✓ Saved wagers to: {wagers_csv_path}")
        
        # Also save latest wagers
        latest_wagers_path = os.path.join(output_dir, "prizepicks_wagers_latest.csv")
        wager_df.to_csv(latest_wagers_path, index=False)
        
        # Save detailed summary
        summary_path = os.path.join(output_dir, f"prizepicks_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("PrizePicks Wager Analysis\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            f.write("OVERALL SUMMARY\n")
            f.write("-"*30 + "\n")
            for key, value in summary.items():
                if isinstance(value, float) and ('rate' in key or 'roi' in key):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}%\n")
                elif isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: ${value:,.2f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n\nBY SPORT\n")
            f.write("-"*30 + "\n")
            f.write(by_sport.to_string())
            
            f.write("\n\nBY PICK TYPE\n")
            f.write("-"*30 + "\n")
            f.write(by_pick_type.to_string())
            
            f.write("\n\nRECENT PREDICTIONS\n")
            f.write("-"*30 + "\n")
            for _, row in df.head(10).iterrows():
                f.write(f"{row['date']} - {row['sport']} - {row.get('player_name', 'Unknown')} - ")
                f.write(f"{row.get('stat_type', 'Unknown')} {row.get('wager_type', '')} {row.get('line_score', '')} - ")
                f.write(f"Result: {row.get('prediction_result', 'Unknown')}\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()