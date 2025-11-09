#!/usr/bin/env python3
"""
Coverage Gap Analysis Script
Calculates exact statements needed to reach 95% coverage for health economics modules
"""

def calculate_coverage_gap(total_statements, current_coverage_pct, target_coverage_pct=95):
    """Calculate statements needed to reach target coverage"""
    covered_statements = int(total_statements * current_coverage_pct / 100)
    target_covered = int(total_statements * target_coverage_pct / 100)
    statements_needed = target_covered - covered_statements
    
    return {
        'total_statements': total_statements,
        'current_coverage_pct': current_coverage_pct,
        'covered_statements': covered_statements,
        'target_coverage_pct': target_coverage_pct,
        'target_covered_statements': target_covered,
        'statements_needed': statements_needed,
        'remaining_statements': total_statements - covered_statements
    }

# Clinical Trials Module
print("=== CLINICAL TRIALS MODULE COVERAGE GAP ANALYSIS ===")
ct_results = calculate_coverage_gap(324, 92, 95)
print(f"Total statements: {ct_results['total_statements']}")
print(f"Current coverage: {ct_results['current_coverage_pct']}%")
print(f"Currently covered: {ct_results['covered_statements']} statements") 
print(f"Target coverage: {ct_results['target_coverage_pct']}%")
print(f"Target covered: {ct_results['target_covered_statements']} statements")
print(f"Additional statements needed: {ct_results['statements_needed']}")
print(f"Remaining uncovered: {ct_results['remaining_statements']} statements")
print()

# HTA Integration Module (Current 37%)
print("=== HTA INTEGRATION MODULE COVERAGE GAP ANALYSIS ===")
hta_results = calculate_coverage_gap(323, 37, 95)
print(f"Total statements: {hta_results['total_statements']}")
print(f"Current coverage: {hta_results['current_coverage_pct']}%")
print(f"Currently covered: {hta_results['covered_statements']} statements")
print(f"Target coverage: {hta_results['target_coverage_pct']}%")
print(f"Target covered: {hta_results['target_covered_statements']} statements")
print(f"Additional statements needed: {hta_results['statements_needed']}")
print(f"Remaining uncovered: {hta_results['remaining_statements']} statements")
print()

# Multi-Domain Module (Current 44%)
print("=== MULTI-DOMAIN MODULE COVERAGE GAP ANALYSIS ===")
md_results = calculate_coverage_gap(279, 44, 95)
print(f"Total statements: {md_results['total_statements']}")
print(f"Current coverage: {md_results['current_coverage_pct']}%")
print(f"Currently covered: {md_results['covered_statements']} statements")
print(f"Target coverage: {md_results['target_coverage_pct']}%")
print(f"Target covered: {md_results['target_covered_statements']} statements")
print(f"Additional statements needed: {md_results['statements_needed']}")
print(f"Remaining uncovered: {md_results['remaining_statements']} statements")
print()

# Summary
print("=== SUMMARY ===")
total_needed = ct_results['statements_needed'] + hta_results['statements_needed'] + md_results['statements_needed']
print(f"Total additional statements needed across all modules: {total_needed}")

print("\n=== PRIORITY RECOMMENDATIONS ===")
if ct_results['statements_needed'] <= 20:
    print("1. HIGH PRIORITY: Clinical Trials - Only ~3 statements needed for 95% coverage")
    
print("2. MEDIUM PRIORITY: HTA Integration - ~187 statements needed")
print("   - Focus on regulatory framework (185-291) and decision-making (328-364) scenarios")

print("3. MEDIUM PRIORITY: Multi-Domain - ~142 statements needed")
print("   - Focus on cross-domain integration (201-216, 254-284) and optimization (333-361)")

print("\n=== COVERAGE CALCULATION DISCREPANCY ALERT ===")
print("The HTA Integration and Multi-Domain modules show coverage decreases")
print("from the baseline. This needs investigation before proceeding with")
print("additional test development to ensure accurate measurement.")