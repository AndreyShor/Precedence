#!/usr/bin/env python3
"""
Analysis script for ablation study results
Matches the actual output format from ablation_study.py
"""

import pandas as pd
import numpy as np

def analyze_results(csv_file='ablation_results.csv'):
    """Analyze ablation study results"""
    
    try:
        # Load results with correct column names
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} results from {csv_file}")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Make sure you ran ablation_study.py first.")
        return
    
    print("=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)
    
    # Group by environment
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        baseline_row = env_data[env_data['Agent'] == 'Baseline']
        
        if baseline_row.empty:
            print(f"Warning: No baseline found for {env}")
            continue
            
        baseline = baseline_row.iloc[0]
        
        print(f"\n{env} RESULTS")
        print("=" * 60)
        
        # Calculate improvements over baseline
        env_data['RewardImprovement'] = env_data['MeanReward'] - baseline['MeanReward']
        env_data['RewardImprovement%'] = (env_data['RewardImprovement'] / abs(baseline['MeanReward'])) * 100
        env_data['FailureReduction'] = baseline['MeanFailures'] - env_data['MeanFailures'] 
        env_data['FailureReduction%'] = (env_data['FailureReduction'] / baseline['MeanFailures']) * 100 if baseline['MeanFailures'] > 0 else 0
        env_data['VarianceReduction%'] = ((baseline['StdReward'] - env_data['StdReward']) / baseline['StdReward']) * 100
        
        # Sort by reward improvement (best first)
        env_data = env_data.sort_values('RewardImprovement', ascending=False)
        
        # Print detailed comparison table
        print(f"{'Agent':<15} {'Reward':<15} {'Δ Reward':<10} {'Δ%':<8} {'Failures':<10} {'Δ Fail%':<10} {'Rollbacks':<10}")
        print("-" * 85)
        
        for _, row in env_data.iterrows():
            agent_name = row['Agent'][:15]
            reward = f"{row['MeanReward']:.1f}±{row['StdReward']:.1f}"
            delta_reward = f"{row['RewardImprovement']:+.1f}"
            delta_pct = f"{row['RewardImprovement%']:+.1f}%"
            failures = f"{row['MeanFailures']:.3f}"
            fail_reduction_pct = f"{row['FailureReduction%']:+.1f}%" if row['FailureReduction%'] != 0 else "n/a"
            rollbacks = f"{row['MeanRollbacks']:.1f}" if row['MeanRollbacks'] > 0 else "n/a"
            
            print(f"{agent_name:<15} {reward:<15} {delta_reward:<10} {delta_pct:<8} {failures:<10} {fail_reduction_pct:<10} {rollbacks:<10}")

def component_contribution_analysis(csv_file='ablation_results.csv'):
    """Analyze individual component contributions"""
    
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*80}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        baseline_row = env_data[env_data['Agent'] == 'Baseline']
        full_model_row = env_data[env_data['Agent'] == 'FullModel']
        
        if baseline_row.empty or full_model_row.empty:
            print(f"Skipping {env} - missing baseline or full model")
            continue
            
        baseline = baseline_row.iloc[0]
        full_model = full_model_row.iloc[0]
        
        print(f"\n{env} - Component Effectiveness Analysis:")
        print("-" * 50)
        print(f"Baseline Performance:    {baseline['MeanReward']:.1f} reward, {baseline['MeanFailures']:.3f} failures")
        print(f"Full Model Performance:  {full_model['MeanReward']:.1f} reward, {full_model['MeanFailures']:.3f} failures")
        print(f"Full Model Improvement:  {full_model['MeanReward'] - baseline['MeanReward']:+.1f} reward ({((full_model['MeanReward'] - baseline['MeanReward']) / abs(baseline['MeanReward'])) * 100:+.1f}%)")
        print()
        
        # Component mapping
        component_mapping = {
            'RollbackOnly': 'Rollback Mechanism Only',
            'PrecedenceOnly': 'Precedence Estimation Only', 
            'ThresholdOnly': 'Threshold Penalty Only',
            'FullModel': 'Complete Algorithm (All Components)'
        }
        
        print("Individual Component Performance:")
        for agent_name, description in component_mapping.items():
            agent_row = env_data[env_data['Agent'] == agent_name]
            if not agent_row.empty:
                agent_data = agent_row.iloc[0]
                improvement = agent_data['MeanReward'] - baseline['MeanReward']
                improvement_pct = (improvement / abs(baseline['MeanReward'])) * 100
                
                # Calculate what percentage of full model improvement this represents
                full_improvement = full_model['MeanReward'] - baseline['MeanReward']
                pct_of_full = (improvement / full_improvement) * 100 if full_improvement != 0 else 0
                
                failure_reduction = baseline['MeanFailures'] - agent_data['MeanFailures']
                failure_reduction_pct = (failure_reduction / baseline['MeanFailures']) * 100 if baseline['MeanFailures'] > 0 else 0
                
                print(f"  {description:<35}: {improvement:+7.1f} reward ({improvement_pct:+5.1f}%) [{pct_of_full:5.1f}% of full model]")
                print(f"    {'':35}   Failure reduction: {failure_reduction:+5.3f} ({failure_reduction_pct:+5.1f}%)")
                if agent_data['MeanRollbacks'] > 0:
                    print(f"    {'':35}   Rollbacks per episode: {agent_data['MeanRollbacks']:.1f}")
                print()

def critical_insights(csv_file='ablation_results.csv'):
    """Generate critical insights for manuscript revision"""
    
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*80}")
    print("CRITICAL INSIGHTS FOR MANUSCRIPT")
    print(f"{'='*80}")
    
    insights = []
    
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        baseline = env_data[env_data['Agent'] == 'Baseline'].iloc[0]
        
        # Sort by performance
        env_data_sorted = env_data.sort_values('MeanReward', ascending=False)
        
        # Find best performer
        best_agent = env_data_sorted.iloc[0]
        
        if best_agent['Agent'] != 'FullModel':
            insights.append(f"⚠️  CRITICAL: In {env}, {best_agent['Agent']} outperforms FullModel")
            insights.append(f"    {best_agent['Agent']}: {best_agent['MeanReward']:.1f} vs FullModel: {env_data[env_data['Agent'] == 'FullModel']['MeanReward'].iloc[0]:.1f}")
        
        # Check if precedence alone helps
        precedence_only = env_data[env_data['Agent'] == 'PrecedenceOnly']
        if not precedence_only.empty:
            prec_improvement = precedence_only.iloc[0]['MeanReward'] - baseline['MeanReward']
            if prec_improvement < 0:
                insights.append(f"⚠️  CRITICAL: In {env}, PrecedenceOnly performs WORSE than baseline ({prec_improvement:.1f})")
        
        # Check rollback efficiency vs performance
        rollback_only = env_data[env_data['Agent'] == 'RollbackOnly']
        full_model = env_data[env_data['Agent'] == 'FullModel']
        if not rollback_only.empty and not full_model.empty:
            ro_perf = rollback_only.iloc[0]['MeanReward']
            fm_perf = full_model.iloc[0]['MeanReward']
            ro_rollbacks = rollback_only.iloc[0]['MeanRollbacks']
            fm_rollbacks = full_model.iloc[0]['MeanRollbacks']
            
            if ro_perf > fm_perf:
                insights.append(f"📊 FINDING: In {env}, RollbackOnly achieves better performance with more rollbacks")
                insights.append(f"    RollbackOnly: {ro_perf:.1f} reward, {ro_rollbacks:.1f} rollbacks")
                insights.append(f"    FullModel: {fm_perf:.1f} reward, {fm_rollbacks:.1f} rollbacks")
                insights.append(f"    Implication: Precedence estimation reduces rollback frequency but hurts performance")
    
    print("\nKey Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR MANUSCRIPT REVISION")
    print(f"{'='*60}")
    
    recommendations = [
        "Consider reframing the paper around rollback mechanism as the core contribution",
        "Investigate why precedence estimation interferes with rollback effectiveness",
        "Test different precedence parameters (λ, K, α_φ) to see if performance can be recovered",
        "Emphasize safety improvements (failure reduction) over performance improvements",
        "Consider positioning precedence estimation as efficiency optimization (fewer rollbacks) rather than performance improvement"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

def generate_manuscript_table(csv_file='ablation_results.csv'):
    """Generate LaTeX table for manuscript"""
    
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*80}")
    print("LATEX TABLES FOR MANUSCRIPT")
    print("="*80)
    
    for env in df['Environment'].unique():
        env_data = df[df['Environment'] == env].copy()
        baseline = env_data[env_data['Agent'] == 'Baseline'].iloc[0]
        
        # Sort by performance
        env_data = env_data.sort_values('MeanReward', ascending=False)
        
        print(f"\n% Table for {env}")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lrrrr}")
        print("\\toprule")
        print("Method & Mean Reward & $\\Delta$ Reward & Failures & Rollbacks \\\\")
        print("\\midrule")
        
        for _, row in env_data.iterrows():
            agent = row['Agent'].replace('_', '\\_')
            reward = f"{row['MeanReward']:.1f}"
            delta = f"{row['MeanReward'] - baseline['MeanReward']:+.1f}" if row['Agent'] != 'Baseline' else "—"
            failures = f"{row['MeanFailures']:.3f}"
            rollbacks = f"{row['MeanRollbacks']:.1f}" if row['MeanRollbacks'] > 0 else "—"
            
            print(f"{agent} & {reward} & {delta} & {failures} & {rollbacks} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        env_caption = env.replace('_', '\\_')
        print(f"\\caption{{Ablation study results for {env_caption}}}")
        print(f"\\label{{tab:ablation_{env.lower().replace('-', '_')}}}")
        print("\\end{table}")

if __name__ == "__main__":
    print("Analyzing  ablation study results...")
    
    # Run all analyses
    analyze_results()
    component_contribution_analysis()
    #critical_insights()
    #generate_manuscript_table()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)