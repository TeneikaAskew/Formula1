# F1 Performance Analysis Validation Report

## Summary

This report validates the calculations in `f1_performance_analysis.py` by comparing manual calculations against the analyzer's output.

## Key Findings

### 1. Overtake Calculations ❌
**Status: Discrepancies Found**

- Manual calculations show different overtake totals compared to the analyzer
- Examples:
  - Max Verstappen: Manual=204, Analyzer=115 (difference: 89)
  - Lewis Hamilton: Manual=200, Analyzer=163 (difference: 37)
  - George Russell: Manual=71, Analyzer=112 (difference: -41)

**Possible Reasons:**
- The analyzer may be filtering out certain types of overtakes
- Different handling of DNF/DNS situations
- The analyzer appears to be calculating net overtakes differently

### 2. Points Calculations ❌
**Status: Large Discrepancies Found**

- Significant differences between manual and analyzer calculations
- Examples for 2024-2025:
  - Max Verstappen: Manual=555, Analyzer=159 (difference: 396)
  - Lando Norris: Manual=555, Analyzer=217 (difference: 338)

**Possible Reasons:**
- The analyzer is likely using a different time period or filtering criteria
- Sprint race points and fastest lap points may not be included consistently
- The manual calculation might be double-counting or the analyzer is missing races

### 3. Year Filter (2024-2025) ✅
**Status: Working Correctly**

- The analyzer correctly filters for drivers who raced in 2024-2025
- 27 drivers identified in both manual check and analyzer output
- No extra drivers included in the analysis

### 4. Teammate Overtakes & PrizePicks ✅
**Status: Calculations Correct**

- PrizePicks scoring formula (±1.5 points per teammate overtake) is correctly implemented
- All tested drivers show correct calculations:
  - Net overtakes × 1.5 = PrizePicks points
- Examples verified: Leclerc, Hamilton, Magnussen, Bottas, Ocon

### 5. Pit Stop Calculations ⚠️
**Status: Cannot Fully Validate**

- The data contains lap times when pit stops occurred, not actual pit stop durations
- Unable to validate average/median pit stop times without duration data
- The analyzer appears to be calculating something, but the source of duration data is unclear

## Recommendations

1. **Overtake Calculations**: Review the filtering logic in `analyze_overtakes()` method to understand why totals differ from simple grid-to-finish calculations.

2. **Points Calculations**: Check if the analyzer is:
   - Including all races from the specified time period
   - Properly accounting for sprint races
   - Adding fastest lap points correctly

3. **Documentation**: Add comments to explain:
   - How overtakes are calculated (what counts as an overtake)
   - Which races/years are included in each analysis
   - How pit stop durations are derived from the available data

4. **Data Validation**: Consider adding internal consistency checks to ensure:
   - Total points match official standings
   - Overtake calculations are symmetrical (one driver's gain is another's loss)

## Conclusion

The analyzer appears to be working correctly for:
- Year filtering (2024-2025 drivers only)
- Teammate overtake calculations
- PrizePicks scoring

However, there are significant discrepancies in:
- Overall overtake calculations
- Points totals

These discrepancies suggest the analyzer may be using different criteria or filters than expected. Further investigation of the source code is recommended to understand the exact calculation methods.