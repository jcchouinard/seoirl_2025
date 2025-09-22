import pandas as pd
import numpy as np
from scipy.stats import pearsonr, t
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import sys

SAMPLE_DATA_FILENAME = './outputs/sample.csv' 
ASSIGNMENTS_FILENAME = './outputs/assignments.csv' 

def load_and_prepare_data(filename=SAMPLE_DATA_FILENAME):
    """
    Loads data from the specified CSV file and prepares it for analysis.
    """
    print(f"Step 1: Loading data from '{filename}'...")
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        print("✅ Data loaded and prepared.\n")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please ensure the CSV file is in the same directory as the script.")
        return None

def find_best_split(df_filtered, num_shuffles=10, outlier_threshold=0.20):
    """
    Performs shuffles, stores all results, and returns them sorted by score.
    """
    print(f"Step 3: Identifying pages with >{outlier_threshold:.0%} outliers to exclude...")
    
    page_stats = {}
    for url, group in df_filtered.groupby('url'):
        q1 = group['clicks'].quantile(0.25)
        q3 = group['clicks'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = group[(group['clicks'] < lower_bound) | (group['clicks'] > upper_bound)]
        outlier_ratio = len(outliers) / len(group)
        page_stats[url] = outlier_ratio

    unused_urls = {url for url, ratio in page_stats.items() if ratio > outlier_threshold}
    usable_urls = list(set(df_filtered['url']) - unused_urls)
    
    print(f"✅ Found {len(unused_urls)} pages to mark as 'Unused'.")
    print(f"✅ Proceeding with {len(usable_urls)} usable pages.\n")

    print(f"Step 4: Performing {num_shuffles} shuffles to find the best Test/Control split...")
    
    all_splits = [] 

    for i in range(num_shuffles):
        np.random.shuffle(usable_urls)
        midpoint = len(usable_urls) // 2
        test_urls = usable_urls[:midpoint]
        control_urls = usable_urls[midpoint:]

        test_df = df_filtered[df_filtered['url'].isin(test_urls)]
        control_df = df_filtered[df_filtered['url'].isin(control_urls)]
        
        test_ts = test_df.groupby('date')['clicks'].sum()
        control_ts = control_df.groupby('date')['clicks'].sum()
        
        def impute_outliers_with_mean(series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            series_mean = series.mean()
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return series.where((series >= lower_bound) & (series <= upper_bound), series_mean)

        test_ts_smoothed = impute_outliers_with_mean(test_ts)
        control_ts_smoothed = impute_outliers_with_mean(control_ts)

        common_index = test_ts_smoothed.index.intersection(control_ts_smoothed.index)
        if len(common_index) > 1:
            corr, _ = pearsonr(test_ts_smoothed[common_index], control_ts_smoothed[common_index])
            print(f"  Shuffle {i+1}/{num_shuffles}: Similarity (Pearson correlation) = {corr:.4f}")
            
            all_splits.append({
                'score': corr,
                'test_urls': test_urls,
                'control_urls': control_urls
            })
        else:
            print(f"  Shuffle {i+1}/{num_shuffles}: Not enough data to compare.")
    
    all_splits.sort(key=lambda x: x['score'], reverse=True)
    
    if all_splits:
        print(f"✅ Found best split with a correlation of {all_splits[0]['score']:.4f}.\n")
    
    return all_splits, unused_urls

def generate_assessment_and_table(best_split, unused_urls, all_urls_df):
    """
    Generates the final assessment report and the assignment table.
    """
    print("Step 5: Generating final outputs for the top-ranked split...")
    
    correlation = best_split['score']
    
    if correlation > 0.95:
        correlation_assessment = "Excellent. The groups are highly correlated and suitable for testing."
    elif correlation > 0.90:
        correlation_assessment = "Good. The groups are well-correlated."
    else:
        correlation_assessment = "Fair. The groups have a moderate correlation. Be cautious with results."

    alpha, power = 0.05, 0.80
    test_ts = all_urls_df[all_urls_df['url'].isin(best_split['test_urls'])].groupby('date')['clicks'].sum()
    control_ts = all_urls_df[all_urls_df['url'].isin(best_split['control_urls'])].groupby('date')['clicks'].sum()
    
    pooled_std = np.sqrt((np.var(test_ts) + np.var(control_ts)) / 2)
    n = len(test_ts)
    t_alpha = t.ppf(1 - alpha / 2, df=2 * n - 2)
    t_beta = t.ppf(power, df=2 * n - 2)
    
    mde_absolute = (t_alpha + t_beta) * pooled_std * np.sqrt(2 / n)
    mde_relative = mde_absolute / control_ts.mean()
    
    verdict = "The experimental setup is viable." if correlation > 0.90 else "The experimental setup is risky. The low correlation may obscure results."

    print("--- OUTPUT 1: Split Quality Assessment ---")
    print(f"📊 Correlation Score: {correlation:.4f}")
    print(f"   Assessment: {correlation_assessment}\n")
    print(f"📈 Minimum Detectable Effect (MDE):")
    print(f"   With 80% power and 95% confidence, you would need to see a lift of at least {mde_relative:.2%} to declare a statistically significant result.")
    print(f"   (This corresponds to an average daily click change of ~{mde_absolute:.0f} clicks.)\n")
    print(f"✅ Final Verdict: {verdict}")
    print("------------------------------------------\n")
    
    all_urls = list(all_urls_df['url'].unique())
    assignments = []
    for url in all_urls:
        if url in best_split['test_urls']: group = "Test"
        elif url in best_split['control_urls']: group = "Control"
        else: group = "Unused"
        assignments.append({'URL': url, 'Group': group})
        
    assignment_df = pd.DataFrame(assignments)
    
    print("--- OUTPUT 2: Assignment Table ---")
    print(assignment_df.to_string())
    print("------------------------------------\n")

    return assignment_df

def plot_top_n_shuffles(all_splits, df_filtered, n=3):
    """
    Creates side-by-side plots for the top n shuffles.
    """
    print(f"Step 6: Generating side-by-side comparison for top {n} splits...")
    
    n = min(n, len(all_splits)) 
    if n == 0:
        print("No splits to plot.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n), sharex=True)
    if n == 1: 
        axes = [axes]

    for i in range(n):
        split_data = all_splits[i]
        ax = axes[i]
        
        test_ts = df_filtered[df_filtered['url'].isin(split_data['test_urls'])].groupby('date')['clicks'].sum().rolling(window=7).mean()
        control_ts = df_filtered[df_filtered['url'].isin(split_data['control_urls'])].groupby('date')['clicks'].sum().rolling(window=7).mean()
        
        ax.plot(test_ts.index, test_ts.values, label='Test Group', color='#007acc', linewidth=2)
        ax.plot(control_ts.index, control_ts.values, label='Control Group', color='#ff6347', linewidth=2)
        
        ax.set_title(f"Rank {i+1}: Correlation = {split_data['score']:.4f}", fontsize=14)
        ax.set_ylabel("7-Day Rolling Avg Clicks")
        ax.legend()
        ax.grid(True)

    plt.xlabel("Date", fontsize=12)
    fig.suptitle("Top N Shuffle Comparisons", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_final_split(best_split, unused_urls, df_filtered):
    """
    Generates a detailed plot for the top-ranked split, including the unused group.
    """
    print("Step 7: Generating detailed plot for the top-ranked split...")

    test_urls, control_urls = best_split['test_urls'], best_split['control_urls']
    test_ts_rolling = df_filtered[df_filtered['url'].isin(test_urls)].groupby('date')['clicks'].sum().rolling(window=7).mean()
    control_ts_rolling = df_filtered[df_filtered['url'].isin(control_urls)].groupby('date')['clicks'].sum().rolling(window=7).mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(test_ts_rolling.index, test_ts_rolling.values, label='Test Group (7-Day Rolling Avg)', color='#007acc', linewidth=2)
    ax.plot(control_ts_rolling.index, control_ts_rolling.values, label='Control Group (7-Day Rolling Avg)', color='#ff6347', linewidth=2)
    
    if unused_urls:
        unused_ts_rolling = df_filtered[df_filtered['url'].isin(unused_urls)].groupby('date')['clicks'].sum().rolling(window=7).mean()
        ax.plot(unused_ts_rolling.index, unused_ts_rolling.values, label='Unused Group (7-Day Rolling Avg)', color='grey', linestyle=':', linewidth=1.5)

    ax.set_title(f'Top-Ranked Split: 7-Day Rolling Average (Correlation: {best_split["score"]:.4f})', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Daily Clicks', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def plot_relative_performance(best_split, df_filtered):
    """
    Generates a relative performance plot for the top-ranked split.
    """
    print("Step 8: Generating relative performance plot for the top-ranked split...")

    test_urls, control_urls = best_split['test_urls'], best_split['control_urls']
    test_ts_rolling = df_filtered[df_filtered['url'].isin(test_urls)].groupby('date')['clicks'].sum().rolling(window=7).mean()
    control_ts_rolling = df_filtered[df_filtered['url'].isin(control_urls)].groupby('date')['clicks'].sum().rolling(window=7).mean()

    relative_performance = ((test_ts_rolling - control_ts_rolling) / control_ts_rolling) * 100
    relative_performance.replace([np.inf, -np.inf], np.nan, inplace=True)
    relative_performance.dropna(inplace=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(relative_performance.index, relative_performance.values, color='black', linewidth=1.5)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.fill_between(relative_performance.index, relative_performance.values, 0, where=relative_performance >= 0, facecolor='green', alpha=0.3)
    ax.fill_between(relative_performance.index, relative_performance.values, 0, where=relative_performance < 0, facecolor='red', alpha=0.3)
    ax.set_title('Top-Ranked Split: Test Group Relative Performance', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Performance Difference (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def save_assignments_to_csv(assignment_df, filename=ASSIGNMENTS_FILENAME):
    """
    Saves the final assignment DataFrame to a CSV file.
    """
    try:
        assignment_df.to_csv(filename, index=False)
        print(f"✅ Assignment table saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving assignment file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    log_filename = './outputs/analysis_log.txt'
    
    with open(log_filename, 'w') as log_file:
        original_stdout = sys.stdout
        sys.stdout = log_file
        
        raw_df = load_and_prepare_data()
        
        plot_data = None
        
        if raw_df is not None:
            print("Step 2: Filtering data to the last 2 years based on the file's date range...")
            
            # --- UPDATED: Calculate date range based on the data file ---
            # Find the most recent date in the dataset
            latest_date_in_file = raw_df['date'].max()
            # Calculate the start date for the filter (2 years before the latest date)
            start_date_for_filter = latest_date_in_file - timedelta(days=730)
            
            # Filter the DataFrame using the date range from the file
            df_filtered = raw_df[raw_df['date'] >= start_date_for_filter].copy()
            
            print(f"✅ Data filtered. Using records from {start_date_for_filter.date()} to {latest_date_in_file.date()}.\n")
            
            all_splits_info, unused_urls_list = find_best_split(df_filtered)
            
            if all_splits_info:
                best_split_info = all_splits_info[0]
                
                final_table = generate_assessment_and_table(best_split_info, unused_urls_list, raw_df)
                save_assignments_to_csv(final_table)
                
                plot_data = {
                    'all_splits': all_splits_info,
                    'best_split': best_split_info, 
                    'unused_urls': unused_urls_list, 
                    'df_filtered': df_filtered
                }
            else:
                print("Could not find any suitable splits. Please check the input data.")
        
        sys.stdout = original_stdout

    print(f"✅ Analysis complete. Log saved to '{log_filename}'.")

    if plot_data:
        plot_top_n_shuffles(plot_data['all_splits'], plot_data['df_filtered'], n=3)
        
        plot_final_split(plot_data['best_split'], plot_data['unused_urls'], plot_data['df_filtered'])
        plot_relative_performance(plot_data['best_split'], plot_data['df_filtered'])