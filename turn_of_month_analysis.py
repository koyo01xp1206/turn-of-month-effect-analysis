#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn-of-Month Effect Analysis
============================
Tests whether buying stocks at month-end and selling at month-start is profitable.
Strategy: Buy on last trading day of month, sell after first few trading days.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import scipy.stats as stats
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
TICKERS = {
    'US_Large': ['SPY', 'QQQ', 'DIA', 'IWM'],
    'US_Sectors': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI'],
    'International': ['EWJ', 'EWZ', 'EWG', 'EWU', 'FXI']
}

YEARS_BACK = 15
HOLDING_PERIODS = [1, 2, 3, 5]  # Days to hold after month-end purchase

def download_data(ticker, years=YEARS_BACK):
    """Download and clean stock data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"  Downloading {ticker}...", end=" ")
        data = yf.download(ticker, start=start_date, progress=False, timeout=15)
        
        if data.empty:
            print("❌ No data")
            return None
            
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Clean and validate
        required_cols = ['Open', 'Close', 'High', 'Low']
        if not all(col in data.columns for col in required_cols):
            print("❌ Missing required columns")
            return None
            
        data = data.dropna()
        if len(data) < 500:  # Need reasonable amount of data
            print("❌ Insufficient data")
            return None
            
        print(f"✅ {len(data)} days")
        return data
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        return None

def identify_month_end_dates(data):
    """Identify last trading day of each month"""
    data = data.copy()
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['YearMonth'] = data['Year'] * 100 + data['Month']
    
    # Find last trading day of each month
    month_ends = data.groupby('YearMonth').tail(1)
    return month_ends.index.tolist()

def identify_month_start_dates(data, days_from_start=3):
    """Identify first few trading days of each month"""
    data = data.copy()
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['YearMonth'] = data['Year'] * 100 + data['Month']
    
    month_starts = []
    for yearmonth, group in data.groupby('YearMonth'):
        # Get first N trading days of month
        first_days = group.head(days_from_start)
        month_starts.extend(first_days.index.tolist())
    
    return month_starts

def calculate_turn_of_month_returns(data, holding_period=3):
    """
    Calculate returns from buying at month-end and selling after holding_period days
    """
    month_end_dates = identify_month_end_dates(data)
    
    tom_returns = []
    tom_trades = []
    
    for end_date in month_end_dates:
        try:
            # Get month-end close price
            if end_date not in data.index:
                continue
                
            end_close = data.loc[end_date, 'Close']
            
            # Find the selling date (holding_period trading days later)
            end_loc = data.index.get_loc(end_date)
            sell_loc = end_loc + holding_period
            
            if sell_loc >= len(data):
                continue
                
            sell_date = data.index[sell_loc]
            sell_close = data.loc[sell_date, 'Close']
            
            # Calculate return
            tom_return = (sell_close - end_close) / end_close
            
            tom_returns.append(tom_return)
            tom_trades.append({
                'buy_date': end_date,
                'sell_date': sell_date,
                'buy_price': end_close,
                'sell_price': sell_close,
                'return': tom_return,
                'holding_days': holding_period
            })
            
        except Exception:
            continue
    
    return pd.Series(tom_returns), pd.DataFrame(tom_trades)

def calculate_random_baseline(data, num_trades, holding_period=3):
    """Calculate baseline returns from random entry points"""
    np.random.seed(42)  # For reproducibility
    
    # Get possible entry dates (exclude last few days to allow for holding period)
    possible_entries = data.index[:-holding_period]
    
    # Sample random entry dates
    random_entries = np.random.choice(possible_entries, size=min(num_trades, len(possible_entries)), replace=False)
    
    baseline_returns = []
    
    for entry_date in random_entries:
        try:
            entry_close = data.loc[entry_date, 'Close']
            
            # Find exit date
            entry_loc = data.index.get_loc(entry_date)
            exit_loc = entry_loc + holding_period
            
            if exit_loc >= len(data):
                continue
                
            exit_date = data.index[exit_loc]
            exit_close = data.loc[exit_date, 'Close']
            
            baseline_return = (exit_close - entry_close) / entry_close
            baseline_returns.append(baseline_return)
            
        except Exception:
            continue
    
    return pd.Series(baseline_returns)

def analyze_turn_of_month_by_holding_period(data, ticker):
    """Analyze turn-of-month effect for different holding periods"""
    results = {}
    
    for holding_period in HOLDING_PERIODS:
        print(f"    Testing {holding_period}-day holding period...")
        
        # Calculate turn-of-month returns
        tom_returns, tom_trades = calculate_turn_of_month_returns(data, holding_period)
        
        if len(tom_returns) < 10:  # Need minimum trades
            continue
            
        # Calculate random baseline
        baseline_returns = calculate_random_baseline(data, len(tom_returns), holding_period)
        
        # Statistics
        results[holding_period] = {
            'tom_returns': tom_returns,
            'baseline_returns': baseline_returns,
            'tom_trades': tom_trades,
            'tom_mean': tom_returns.mean(),
            'baseline_mean': baseline_returns.mean(),
            'tom_std': tom_returns.std(),
            'baseline_std': baseline_returns.std(),
            'tom_win_rate': (tom_returns > 0).mean(),
            'baseline_win_rate': (baseline_returns > 0).mean(),
            'num_trades': len(tom_returns)
        }
        
        # Statistical test
        if len(tom_returns) >= 2 and len(baseline_returns) >= 2:
            t_stat, p_val = stats.ttest_ind(tom_returns, baseline_returns, equal_var=False)
            results[holding_period]['t_stat'] = t_stat
            results[holding_period]['p_value'] = p_val
        else:
            results[holding_period]['t_stat'] = np.nan
            results[holding_period]['p_value'] = np.nan
    
    return results

def analyze_monthly_patterns(data, holding_period=3):
    """Analyze which months show strongest turn-of-month effect"""
    month_end_dates = identify_month_end_dates(data)
    
    monthly_returns = {month: [] for month in range(1, 13)}
    
    for end_date in month_end_dates:
        try:
            month = end_date.month
            
            end_close = data.loc[end_date, 'Close']
            end_loc = data.index.get_loc(end_date)
            sell_loc = end_loc + holding_period
            
            if sell_loc >= len(data):
                continue
                
            sell_date = data.index[sell_loc]
            sell_close = data.loc[sell_date, 'Close']
            
            tom_return = (sell_close - end_close) / end_close
            monthly_returns[month].append(tom_return)
            
        except Exception:
            continue
    
    # Calculate monthly statistics
    monthly_stats = {}
    for month in range(1, 13):
        if len(monthly_returns[month]) > 0:
            returns = pd.Series(monthly_returns[month])
            monthly_stats[month] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'win_rate': (returns > 0).mean(),
                'count': len(returns)
            }
    
    return monthly_stats

def create_comprehensive_charts(results_dict, ticker, output_dir):
    """Create comprehensive visualization charts"""
    
    # Chart 1: Holding Period Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{ticker}: Turn-of-Month Effect Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    holding_periods = []
    tom_means = []
    baseline_means = []
    tom_win_rates = []
    baseline_win_rates = []
    
    for period, results in results_dict.items():
        holding_periods.append(period)
        tom_means.append(results['tom_mean'] * 100)
        baseline_means.append(results['baseline_mean'] * 100)
        tom_win_rates.append(results['tom_win_rate'] * 100)
        baseline_win_rates.append(results['baseline_win_rate'] * 100)
    
    # Mean returns comparison
    x = np.arange(len(holding_periods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, tom_means, width, label='Turn-of-Month', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width/2, baseline_means, width, label='Random Baseline', color='blue', alpha=0.7)
    
    ax1.set_title('Mean Returns by Holding Period')
    ax1.set_xlabel('Holding Period (Days)')
    ax1.set_ylabel('Mean Return (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(holding_periods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}%', ha='center', va='bottom', fontsize=9)
    
    # Win rate comparison
    bars3 = ax2.bar(x - width/2, tom_win_rates, width, label='Turn-of-Month', color='orange', alpha=0.7)
    bars4 = ax2.bar(x + width/2, baseline_win_rates, width, label='Random Baseline', color='lightblue', alpha=0.7)
    
    ax2.set_title('Win Rate by Holding Period')
    ax2.set_xlabel('Holding Period (Days)')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(holding_periods)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Return distribution for best performing holding period
    if results_dict:
        best_period = max(results_dict.keys(), key=lambda k: results_dict[k]['tom_mean'])
        best_results = results_dict[best_period]
        
        ax3.hist(best_results['tom_returns'] * 100, bins=20, alpha=0.7, color='red', 
                label=f'Turn-of-Month ({best_period}d)', density=True)
        ax3.hist(best_results['baseline_returns'] * 100, bins=20, alpha=0.7, color='blue',
                label=f'Random Baseline ({best_period}d)', density=True)
        ax3.set_title(f'Return Distribution (Best: {best_period}-day holding)')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Cumulative performance for best period
    if results_dict:
        tom_cumulative = (1 + best_results['tom_returns']).cumprod()
        baseline_cumulative = (1 + best_results['baseline_returns']).cumprod()
        
        ax4.plot(range(len(tom_cumulative)), tom_cumulative.values, 
                color='red', linewidth=2, label=f'Turn-of-Month ({best_period}d)')
        ax4.plot(range(len(baseline_cumulative)), baseline_cumulative.values,
                color='blue', linewidth=2, label=f'Random Baseline ({best_period}d)')
        ax4.set_title(f'Cumulative Performance ({best_period}-day holding)')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ticker}_turn_of_month_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_monthly_pattern_chart(monthly_stats, ticker, output_dir):
    """Create chart showing turn-of-month effect by calendar month"""
    if not monthly_stats:
        return
        
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    means = []
    win_rates = []
    counts = []
    
    for month in range(1, 13):
        if month in monthly_stats:
            means.append(monthly_stats[month]['mean'] * 100)
            win_rates.append(monthly_stats[month]['win_rate'] * 100)
            counts.append(monthly_stats[month]['count'])
        else:
            means.append(0)
            win_rates.append(50)
            counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{ticker}: Turn-of-Month Effect by Calendar Month', fontsize=14, fontweight='bold')
    
    # Monthly mean returns
    colors = ['red' if m < 0 else 'green' for m in means]
    bars1 = ax1.bar(months, means, color=colors, alpha=0.7)
    ax1.set_title('Mean Turn-of-Month Returns by Month')
    ax1.set_ylabel('Mean Return (%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean in zip(bars1, means):
        if mean != 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{mean:.3f}%', ha='center', va='bottom', fontsize=9)
    
    # Monthly win rates
    colors2 = ['red' if w < 50 else 'green' for w in win_rates]
    bars2 = ax2.bar(months, win_rates, color=colors2, alpha=0.7)
    ax2.set_title('Win Rate by Month')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, win_rate, count in zip(bars2, win_rates, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{win_rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ticker}_monthly_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()

def format_results_summary(results_dict, ticker):
    """Format results for easy reading"""
    summary = f"\n📊 {ticker} Turn-of-Month Analysis Results:\n"
    summary += "=" * 50 + "\n"
    
    for period, results in results_dict.items():
        tom_mean = results['tom_mean'] * 100
        baseline_mean = results['baseline_mean'] * 100
        difference = tom_mean - baseline_mean
        
        summary += f"\n{period}-Day Holding Period:\n"
        summary += f"  Turn-of-Month: {tom_mean:.3f}% (Win Rate: {results['tom_win_rate']*100:.1f}%)\n"
        summary += f"  Random Baseline: {baseline_mean:.3f}% (Win Rate: {results['baseline_win_rate']*100:.1f}%)\n"
        summary += f"  Difference: {difference:.3f}% per trade\n"
        summary += f"  Trades: {results['num_trades']}\n"
        
        if not np.isnan(results['p_value']):
            significance = "✅ Significant" if results['p_value'] < 0.05 else "❌ Not Significant"
            summary += f"  Statistical Test: t={results['t_stat']:.3f}, p={results['p_value']:.4f} ({significance})\n"
    
    return summary

def main():
    print("🔄 Turn-of-Month Effect Analysis")
    print("=" * 50)
    print("Strategy: Buy last trading day of month, sell after N days")
    print(f"Testing holding periods: {HOLDING_PERIODS} days")
    print()
    
    # Create output directory
    output_dir = "turn_of_month_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    summary_data = []
    
    # Test each category
    for category, tickers in TICKERS.items():
        print(f"\n📈 Testing {category} Category:")
        print("-" * 30)
        
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}:")
            
            # Download data
            data = download_data(ticker)
            if data is None:
                continue
            
            # Analyze turn-of-month effect
            results = analyze_turn_of_month_by_holding_period(data, ticker)
            if not results:
                print(f"  ❌ Insufficient data for analysis")
                continue
            
            all_results[ticker] = results
            
            # Print summary
            summary = format_results_summary(results, ticker)
            print(summary)
            
            # Create charts
            create_comprehensive_charts(results, ticker, output_dir)
            
            # Analyze monthly patterns
            monthly_stats = analyze_monthly_patterns(data)
            create_monthly_pattern_chart(monthly_stats, ticker, output_dir)
            
            # Save detailed results
            for period, result in results.items():
                summary_data.append({
                    'Ticker': ticker,
                    'Category': category,
                    'Holding_Period': period,
                    'TOM_Return': result['tom_mean'],
                    'Baseline_Return': result['baseline_mean'],
                    'Difference': result['tom_mean'] - result['baseline_mean'],
                    'TOM_WinRate': result['tom_win_rate'],
                    'Baseline_WinRate': result['baseline_win_rate'],
                    'Num_Trades': result['num_trades'],
                    'P_Value': result['p_value']
                })
    
    # Create summary analysis
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/turn_of_month_summary.csv", index=False)
        
        # Overall conclusion
        print("\n" + "=" * 60)
        print("🎯 OVERALL CONCLUSIONS")
        print("=" * 60)
        
        # Find best performing strategies
        best_strategies = summary_df.nlargest(3, 'Difference')
        worst_strategies = summary_df.nsmallest(3, 'Difference')
        
        print("\n🏆 Best Performing Turn-of-Month Strategies:")
        for _, row in best_strategies.iterrows():
            print(f"  {row['Ticker']} ({row['Holding_Period']}d): +{row['Difference']*100:.3f}% difference")
        
        print("\n📉 Worst Performing Turn-of-Month Strategies:")
        for _, row in worst_strategies.iterrows():
            print(f"  {row['Ticker']} ({row['Holding_Period']}d): {row['Difference']*100:.3f}% difference")
        
        # Statistical significance summary
        significant_count = sum(1 for _, row in summary_df.iterrows() if row['P_Value'] < 0.05)
        total_tests = len(summary_df)
        
        print(f"\n📊 Statistical Significance:")
        print(f"  {significant_count}/{total_tests} strategies show significant turn-of-month effect")
        print(f"  {(significant_count/total_tests)*100:.1f}% of tests are statistically significant")
        
        # Average effect across all strategies
        avg_difference = summary_df['Difference'].mean()
        print(f"\n📈 Average Turn-of-Month Effect: {avg_difference*100:.3f}% per trade")
        
        if avg_difference > 0:
            print("✅ Overall positive turn-of-month effect detected!")
        else:
            print("❌ No significant turn-of-month effect detected overall")
    
    print(f"\n📁 Analysis complete! Results saved to: {output_dir}/")
    print("Files created:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")

if __name__ == "__main__":
    main()