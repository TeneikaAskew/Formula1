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
    
    wagers = []
    predictions = []
    projections = []
    players = []
    
    # Extract included data (predictions, projections, players)
    if 'included' in data:
        for item in data['included']:
            if item['type'] == 'prediction':
                predictions.append(item)
            elif item['type'] == 'projection':
                projections.append(item)
            elif item['type'] == 'new_player':
                players.append(item)
    
    # Create lookup dictionaries
    prediction_map = {p['id']: p for p in predictions}
    projection_map = {p['id']: p for p in projections}
    player_map = {p['id']: p for p in players}
    
    # Process wagers
    for wager in data['data']:
        if wager['type'] != 'new_wager':
            continue
            
        w = wager['attributes']
        wager_id = wager['id']
        
        # Calculate actual win/loss
        is_win = w['result'] in ['won', 'partial_win']
        actual_payout = w['amount_won_cents'] / 100.0
        bet_amount = w['amount_bet_cents'] / 100.0
        potential_payout = w['amount_to_win_cents'] / 100.0
        profit_loss = actual_payout - bet_amount
        
        # Get prediction details
        player_names = []
        player_results = []
        projection_details = []
        
        if 'predictions' in wager['relationships'] and wager['relationships']['predictions']['data']:
            for pred_ref in wager['relationships']['predictions']['data']:
                pred_id = pred_ref['id']
                if pred_id in prediction_map:
                    pred = prediction_map[pred_id]
                    pred_attrs = pred['attributes']
                    
                    # Get projection details
                    if 'projection' in pred['relationships'] and pred['relationships']['projection']['data']:
                        proj_id = pred['relationships']['projection']['data']['id']
                        if proj_id in projection_map:
                            proj = projection_map[proj_id]
                            proj_attrs = proj['attributes']
                            
                            # Get player name
                            player_name = 'Unknown'
                            if 'new_player' in proj['relationships'] and proj['relationships']['new_player']['data']:
                                player_id = proj['relationships']['new_player']['data']['id']
                                if player_id in player_map:
                                    player_name = player_map[player_id]['attributes']['display_name']
                            
                            player_names.append(player_name)
                            
                            # Determine if player won/lost
                            is_over = pred_attrs.get('is_over', None)
                            outcome = pred_attrs.get('outcome', 'unknown')
                            hit_outcome = outcome == 'hit'
                            player_result = 'Won' if hit_outcome else 'Lost'
                            if outcome == 'push':
                                player_result = 'Push'
                            elif outcome == 'unknown':
                                player_result = 'Unknown'
                            player_results.append(f"{player_name}:{player_result}")
                            
                            # Store projection details
                            projection_details.append({
                                'player': player_name,
                                'stat_type': proj_attrs.get('stat_type', ''),
                                'line_score': proj_attrs.get('line_score', ''),
                                'is_over': is_over,
                                'outcome': outcome
                            })
        
        # Create lineup entry
        pick_type = f"{w['parlay_count']}-Pick"
        play_type = 'Power Play' if not w.get('pick_protection', True) else 'Flex Play'
        
        wager_data = {
            'lineup_id': wager_id,
            'date': datetime.fromisoformat(w['created_at'].replace('-04:00', '-04:00')).strftime('%B %d, %Y'),
            'pick_type': pick_type,
            'entry_fee': f"${bet_amount:.2f}",
            'play_type': play_type,
            'payout': f"${potential_payout:.2f}",
            'result': 'Win' if is_win else 'Loss',
            'players': ', '.join(player_names),
            'player_results': '; '.join(player_results),
            'sport': w.get('sport', 'Unknown'),
            'actual_payout': actual_payout,
            'profit_loss': profit_loss,
            'created_at': w['created_at'],
            'updated_at': w['updated_at']
        }
        
        wagers.append(wager_data)
    
    return pd.DataFrame(wagers)

def generate_summary(df):
    """Generate summary statistics"""
    
    # Clean up numeric columns
    df['entry_fee_amount'] = df['entry_fee'].str.replace('$', '').astype(float)
    df['payout_amount'] = df['payout'].str.replace('$', '').astype(float)
    
    total_invested = df['entry_fee_amount'].sum()
    total_payout = df['actual_payout'].sum()
    net_profit = df['profit_loss'].sum()
    
    summary = {
        'total_lineups': len(df),
        'total_wins': len(df[df['result'] == 'Win']),
        'total_losses': len(df[df['result'] == 'Loss']),
        'win_rate': (len(df[df['result'] == 'Win']) / len(df) * 100) if len(df) > 0 else 0,
        'total_invested': total_invested,
        'total_payout': total_payout,
        'net_profit_loss': net_profit,
        'roi': (net_profit / total_invested * 100) if total_invested > 0 else 0
    }
    
    # By sport
    by_sport = df.groupby('sport').agg({
        'lineup_id': 'count',
        'result': lambda x: (x == 'Win').sum(),
        'profit_loss': 'sum',
        'entry_fee_amount': 'sum'
    }).rename(columns={
        'lineup_id': 'total_lineups',
        'result': 'wins',
        'profit_loss': 'net_profit',
        'entry_fee_amount': 'invested'
    })
    by_sport['win_rate'] = (by_sport['wins'] / by_sport['total_lineups'] * 100).round(1)
    by_sport['roi'] = (by_sport['net_profit'] / by_sport['invested'] * 100).round(1)
    
    # By pick type
    by_pick_type = df.groupby('pick_type').agg({
        'lineup_id': 'count',
        'result': lambda x: (x == 'Win').sum(),
        'profit_loss': 'sum'
    }).rename(columns={
        'lineup_id': 'total',
        'result': 'wins'
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
        print(f"\n✓ Parsed {len(df)} wagers")
        
        # Sort by date
        df['created_datetime'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_datetime', ascending=False)
        
        # Generate summary
        summary, by_sport, by_pick_type = generate_summary(df)
        
        # Print summary
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)
        print(f"Total Lineups: {summary['total_lineups']}")
        print(f"Wins: {summary['total_wins']} ({summary['win_rate']:.1f}%)")
        print(f"Losses: {summary['total_losses']}")
        print(f"\nTotal Invested: ${summary['total_invested']:,.2f}")
        print(f"Total Payout: ${summary['total_payout']:,.2f}")
        print(f"Net Profit/Loss: ${summary['net_profit_loss']:,.2f}")
        print(f"ROI: {summary['roi']:.1f}%")
        
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
        csv_path = os.path.join(output_dir, f"prizepicks_wagers_{timestamp}.csv")
        
        # Select columns for CSV
        csv_columns = ['lineup_id', 'date', 'sport', 'pick_type', 'entry_fee', 'play_type', 
                      'payout', 'result', 'actual_payout', 'profit_loss', 'players', 'player_results']
        df[csv_columns].to_csv(csv_path, index=False)
        
        print(f"\n✓ Saved to: {csv_path}")
        
        # Also save latest
        latest_path = os.path.join(output_dir, "prizepicks_wagers_latest.csv")
        df[csv_columns].to_csv(latest_path, index=False)
        
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
            
            f.write("\n\nRECENT LINEUPS\n")
            f.write("-"*30 + "\n")
            for _, row in df.head(10).iterrows():
                f.write(f"{row['date']} - {row['sport']} - {row['pick_type']} - ")
                f.write(f"{row['entry_fee']} - {row['result']} - P/L: ${row['profit_loss']:.2f}\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()