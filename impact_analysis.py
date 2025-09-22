import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os


# Manually set the filenames, intervention date, and uplift for your analysis here
SAMPLE_DATA_FILENAME = './outputs/sample.csv' 
ASSIGNMENTS_FILENAME = './outputs/assignments.csv'
INTERVENTION_DATE = '2025-06-22' # Set the exact start date of the experiment
ARTIFICIAL_UPLIFT = 1.15           # e.g., 1.08 for +8%, 1.0 for no lift


def run_did_analysis(did_df):
    """
    Performs a Difference-in-Differences (DiD) regression analysis.
    """
    print("\nStep 2: Running Difference-in-Differences (DiD) regression...")
    
    model = smf.ols('clicks ~ treated + post + treated:post', data=did_df).fit()
    
    did_coefficient = model.params['treated:post']
    p_value = model.pvalues['treated:post']
    conf_interval = model.conf_int().loc['treated:post']

    baseline_clicks = did_df[(did_df['treated'] == 0) & (did_df['post'] == 1)]['clicks'].mean()
    
    if baseline_clicks > 0:
        relative_lift = (did_coefficient / baseline_clicks) * 100
    else:
        relative_lift = np.inf

    print("\n--- Difference-in-Differences (DiD) Results ---")
    print(f"📊 Estimated Average Lift (Treatment Effect): {did_coefficient:.2f} clicks per day")
    print(f"   - Relative Lift vs. Control Group: {relative_lift:.2f}%")
    print(f"   - 95% Confidence Interval for Lift: [{conf_interval[0]:.2f}, {conf_interval[1]:.2f}] clicks")
    print(f"   - P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✅ The result is statistically significant at the p < 0.05 level.")
    else:
        print("❌ The result is not statistically significant at the p < 0.05 level.")
        
    return relative_lift, model

def plot_did_causal_style(did_df, did_model):
    """
    Creates a three-panel plot with dynamic confidence intervals.
    """
    print("\nStep 3: Generating Causal Impact style three-panel plot with confidence intervals...")

    daily_performance = did_df.pivot_table(index='date', columns='treated', values='clicks')
    daily_performance.columns = ['control', 'test']
    daily_performance_rolling = daily_performance.rolling(window=7).mean().dropna()
    post_period_start_date = did_df[did_df['post'] == 1]['date'].min()
    
    pre_period = daily_performance_rolling[daily_performance_rolling.index < post_period_start_date]
    post_period = daily_performance_rolling[daily_performance_rolling.index >= post_period_start_date]

    baseline_difference = (pre_period['test'] - pre_period['control']).mean()
    counterfactual = post_period['control'] + baseline_difference
    
    pre_period_diff_std = (pre_period['test'] - pre_period['control']).std()
    ci_margin = 1.96 * pre_period_diff_std
    counterfactual_lower = counterfactual - ci_margin
    counterfactual_upper = counterfactual + ci_margin

    pointwise_effect = post_period['test'] - counterfactual
    cumulative_effect = pointwise_effect.cumsum()
    
    effect_stderr = did_model.bse['treated:post']
    days_in_post = np.arange(1, len(post_period) + 1)
    cumulative_stderr = np.sqrt(days_in_post) * effect_stderr
    cumulative_ci_margin = 1.96 * cumulative_stderr
    cumulative_lower = cumulative_effect - cumulative_ci_margin
    cumulative_upper = cumulative_effect + cumulative_ci_margin

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    ax = axes[0]
    ax.plot(daily_performance_rolling['test'], 'k', label='Observed Data (Test)')
    ax.plot(counterfactual, 'r--', label='Counterfactual Prediction')
    ax.fill_between(counterfactual.index, counterfactual_lower, counterfactual_upper, color='red', alpha=0.2, label='95% Confidence Interval')
    ax.axvline(post_period_start_date, color='grey', linestyle='--', linewidth=1.5)
    ax.set_title("Panel 1: Observed Data and Counterfactual Prediction", fontsize=14)
    ax.set_ylabel("Clicks (7-Day Avg)")
    ax.legend()

    ax = axes[1]
    pointwise_lower = pointwise_effect - ci_margin
    pointwise_upper = pointwise_effect + ci_margin
    ax.plot(pointwise_effect, 'r', label='Pointwise Effect')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.5)
    ax.axvline(post_period_start_date, color='grey', linestyle='--', linewidth=1.5)
    ax.fill_between(pointwise_effect.index, pointwise_lower, pointwise_upper, color='red', alpha=0.2, label='95% Confidence Interval')
    ax.set_title("Panel 2: Pointwise Causal Effect", fontsize=14)
    ax.set_ylabel("Difference (Observed - Predicted)")
    ax.legend()
    
    ax = axes[2]
    ax.plot(cumulative_effect, 'r', label='Cumulative Effect')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.5)
    ax.axvline(post_period_start_date, color='grey', linestyle='--', linewidth=1.5)
    ax.fill_between(cumulative_effect.index, cumulative_lower, cumulative_upper, color='red', alpha=0.2, label='95% Confidence Interval')
    ax.set_title("Panel 3: Cumulative Causal Effect", fontsize=14)
    ax.set_ylabel("Cumulative Difference")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    log_filename = os.path.join(output_dir, 'did_analysis_log.txt')
    
    with open(log_filename, 'w') as log_file:
        original_stdout = sys.stdout
        sys.stdout = log_file

        try:
            print("Step 1: Loading and preparing data for DiD analysis...")
            assignments = pd.read_csv(ASSIGNMENTS_FILENAME)
            raw_df = pd.read_csv(SAMPLE_DATA_FILENAME)
            
            raw_df['date'] = pd.to_datetime(raw_df['date'])

            latest_date_in_file = raw_df['date'].max()
            start_date_for_filter = latest_date_in_file - timedelta(days=730)
            df_filtered = raw_df[raw_df['date'] >= start_date_for_filter].copy()

            did_df = df_filtered.merge(assignments, left_on='url', right_on='URL')
            did_df = did_df[did_df['Group'].isin(['Test', 'Control'])]

            post_period_start_date = pd.to_datetime(INTERVENTION_DATE)
            did_df['post'] = (did_df['date'] >= post_period_start_date).astype(int)
            did_df['treated'] = (did_df['Group'] == 'Test').astype(int)
            
            model_df = did_df.groupby(['date', 'treated', 'post'])['clicks'].sum().reset_index()
            
            print("✅ Data prepared successfully.")
            
            if ARTIFICIAL_UPLIFT != 1.0:
                uplift_percentage = (ARTIFICIAL_UPLIFT - 1) * 100
                print(f"\nNOTE: Artificially adding a +{uplift_percentage:.0f}% lift to the test group in the post-period for demonstration.")
                model_df.loc[(model_df['treated'] == 1) & (model_df['post'] == 1), 'clicks'] *= ARTIFICIAL_UPLIFT
            
            lift, model = run_did_analysis(model_df)
            
            sys.stdout = original_stdout
            
            print(f"\n✅ Analysis complete. Log saved to '{log_filename}'.")
            plot_did_causal_style(model_df, model)

        except FileNotFoundError as e:
            sys.stdout = original_stdout
            print(f"Error: Could not find a required file. {e}")
            print(f"Please ensure '{ASSIGNMENTS_FILENAME}' and '{SAMPLE_DATA_FILENAME}' exist in the correct directories.")
        except Exception as e:
            sys.stdout = original_stdout
            print(f"An unexpected error occurred: {e}")