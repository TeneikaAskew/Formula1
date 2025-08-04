# Prize Picks Pattern Analysis Integration

## Overview

The Prize Picks pattern analysis system has been successfully integrated into the F1 ML pipeline. It provides historical performance insights and recommendations based on actual betting results.

## Key Components

### 1. Pattern Analyzer (`f1_prize_picks_insights.py`)
- Analyzes historical performance by prop type, lineup size, driver, and team
- Provides recommendations based on success rates
- Generates comprehensive reports with insights

### 2. Report Generator (`generate_prize_picks_report.py`)
- Creates detailed reports combining portfolio analysis with historical patterns
- Identifies risks and opportunities in current portfolios
- Provides actionable recommendations

## Pattern Analysis Summary

### Prop Type Performance
| Prop Type | Success Rate | ROI Impact | Recommendation |
|-----------|--------------|------------|----------------|
| Starting Position | 100% | +$210 | HIGHLY RECOMMENDED |
| 1st Pit Stop Time | 56% | +$8 | PROCEED WITH CAUTION |
| F1 Points | 0% | -$60 | AVOID |
| Overtake Points | 0% | -$40 | AVOID |
| F1 Sprint Points | 0% | $0 | AVOID |

### Lineup Size Performance
| Size | Win Rate | Total ROI | Recommendation |
|------|----------|-----------|----------------|
| 2-Pick | 33% | +$30 | RECOMMENDED |
| 3-Pick | 0% | -$20 | USE SPARINGLY |
| 4-Pick | 0% | -$40 | AVOID |
| 5-Pick | 33% | +$168 | HIGH VARIANCE |

### Driver Performance
| Driver | Success Rate | Best Prop | Recommendation |
|--------|--------------|-----------|----------------|
| Max Verstappen | 75% | Starting Position | TARGET |
| Charles Leclerc | 67% | Starting Position | TARGET |
| Lando Norris | 50% | Starting Position | SELECTIVE |
| Lewis Hamilton | 25% | Starting Position | SELECTIVE |
| Fernando Alonso | 0% | - | AVOID |

### Team Performance
| Team | Success Rate | Recommendation |
|------|--------------|----------------|
| Red Bull | 75% | TARGET |
| Ferrari | 67% | TARGET |
| McLaren | 50% | SELECTIVE |
| Mercedes | 38% | CAUTION |
| Aston Martin | 0% | AVOID |

## Integration Usage

### 1. Generate Prize Picks Report
```bash
cd notebooks/advanced
python generate_prize_picks_report.py
```

### 2. Use in Python Scripts
```python
from f1_prize_picks_insights import PrizePicksPatternAnalyzer

# Initialize analyzer
analyzer = PrizePicksPatternAnalyzer()

# Get recommendations
prop_rec = analyzer.get_prop_type_recommendation('Starting Position')
driver_rec = analyzer.get_driver_recommendation('Max Verstappen')

# Generate insights report
insights = analyzer.format_insights_for_report()
print(insights)
```

### 3. Integrate with Portfolio Optimization
```python
from f1_prize_picks_insights import add_insights_to_prize_picks_report

# Your portfolio data
portfolio = {...}  # Prize Picks portfolio

# Generate report with insights
report = add_insights_to_prize_picks_report(portfolio, bankroll=1000)
print(report)
```

## Key Insights

### 🟢 Winning Patterns
1. **Starting Position props = 100% success rate** (7/7)
2. **2-pick lineups with starting position = profitable**
3. **Conservative pit stop time overs = 56% success**
4. **Red Bull/Ferrari drivers for starting position**

### 🔴 Losing Patterns
1. **Overtake props = 0% success rate** (0/7)
2. **F1 Points props = 0% success rate** (0/9)
3. **Mercedes drivers for overtakes = avoid**
4. **Aston Martin drivers = avoid completely**
5. **4+ pick lineups = higher variance**

## Recommended Strategy

1. **Focus 80% on Starting Position props**
2. **Use 2-3 pick lineups maximum**
3. **Target Red Bull, Ferrari, McLaren drivers**
4. **Completely avoid Overtake and F1 Points props**
5. **Research qualifying pace on Saturday**

## Next Steps

1. **Track timing of successful picks** (practice vs qualifying analysis)
2. **Note weather conditions for each race**
3. **Add confidence ratings (1-5) for each pick**
4. **Track research time spent per lineup**
5. **Monitor which qualifying positions translate to wins**

## Example Output

```
================================================================================
PRIZE PICKS PATTERN ANALYSIS
================================================================================

📊 Prop Type Performance:
------------------------------------------------------------
Prop Type               Picks     Hits    Success      ROI Impact
------------------------------------------------------------
Starting Position           7        7       100%           $+210
1st Pit Stop Time           9        5        56%             $+8
F1 Points                   9        0         0%            $-60
Overtake Points             7        0         0%            $-40
F1 Sprint Points            5        0         0% $0 (free plays)

⚠️ RISK ASSESSMENT:
------------------------------------------------------------
  ❌ Portfolio contains 3 Overtake props (0% historical success)
  ❌ Only 0% Starting Position props (recommend 80%+)

💡 RECOMMENDATIONS TO IMPROVE PORTFOLIO:
------------------------------------------------------------
  1. Remove all Overtake props
  2. Replace F1 Points props with Starting Position
  3. Increase Starting Position props to 80% of portfolio
```

## Files Created

1. **`f1_prize_picks_insights.py`** - Core pattern analysis module
2. **`generate_prize_picks_report.py`** - Report generation script
3. **`PRIZE_PICKS_INTEGRATION.md`** - This documentation

## Testing

To test the integration:
```bash
# Run pattern analysis
python f1_prize_picks_insights.py

# Generate full report
python generate_prize_picks_report.py
```

Reports are saved to: `pipeline_outputs/prize_picks_report_YYYYMMDD_HHMMSS.txt`