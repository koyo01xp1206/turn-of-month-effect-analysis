#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japan Turn-of-Month Effect Analysis
==================================
Tests turn-of-month effect specifically in Japanese markets using various Japanese assets.
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

# Japanese Market Configuration
JAPANESE_TICKERS = {
    'Japan_Indices': [
        '^N225',    # Nikkei 225 Index
        '^N300',    # Nikkei 300 Index  
        '^TPX',     # TOPIX Index
    ],
    'Japan_ETFs': [
        'EWJ',      # iShares MSCI Japan ETF (US-listed)
        '1321.T',   # Nikkei 225 ETF (Tokyo-listed)
        '1306.T',   # TOPIX ETF (Tokyo-listed)
        'DXJ',      # WisdomTree Japan Hedged Equity Fund
        'HEWJ',     # iShares Currency Hedged MSCI Japan ETF
    ],
    'Japan_Sectors': [
        'EWJ',      # Overall Japan
        'FLEP',     # Japan Consumer Discretionary
        # Note: Limited sector-specific Japan ETFs available
    ],
    'Major_Japanese_Stocks': [
        '7203.T',   # Toyota Motor Corporation
        '6758.T',   # Sony Group Corporation  
        '9984.T',   # SoftBank Group Corp
        '8035.T',   # Tokyo Electron
        '6861.T',   # Keyence Corporation
        '4568.T',   # First Sumitomo Pharma
        '9433.T',   # KDDI Corporation
        '8306.T',   # Mitsubishi UFJ Financial Group
        '9432.T',   # NTT (Nippon Telegraph and Telephone)
        '7974.T',   # Nintendo Co., Ltd.
    ]
}

YEARS_BACK = 15
HOLDING_PERIODS = [1, 2, 3, 5]  # Days to hold after month-end purchase

def download_japanese_data(ticker, years=YEARS_BACK):
    """Download and clean Japanese stock/ETF data with special handling"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"  Downloading {ticker}...", end=" ")
        
        # Special handling for different ticker types
        if ticker.endswith('.T'):
            # Tokyo Stock Exchange - may need different approach
            data = yf.download(ticker, start=start_date, progress=False, timeout=20)
        elif ticker.startswith('^'):
            # Index data
            data = yf.download(ticker, start=start_date, progress=False, timeout=20)
        else:
            # US-listed Japan ETFs
            data = yf.download(ticker, start=start_date, progress=False, timeout=15)
        
        if data.empty:
            print("❌ No data")
            return None
            
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Clean and validate
        required_cols = ['Open', 'Close']
        if not all(col in data.columns for col in required_cols):
            print("❌ Missing required columns")
            return None
            
        data = data.dropna(subset=['Open', 'Close'])
        if len(data) < 500:  # Need reasonable amount of data
            print(f"❌ Insufficient data ({len(data)} days)")
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

def analyze_japanese_market_patterns(data, holding_period=3):
    """Analyze specific patterns in Japanese markets"""
    month_end_dates = identify_month_end_dates(data)
    
    # Monthly patterns
    monthly_returns = {month: [] for month in range(1, 13)}
    
    # Year-end/New Year effect (important in Japan)
    year_end_returns = []
    golden_week_returns = []  # Around May (Golden Week in Japan)
    
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
            
            # Special Japanese periods
            if month == 12:  # December (Year-end)
                year_end_returns.append(tom_return)
            elif month == 4:  # April (End of Japanese fiscal year)
                year_end_returns.append(tom_return)
            elif month == 5:  # May (Golden Week effect)
                golden_week_returns.append(tom_return)
            
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
    
    # Japanese special periods
    special_periods = {
        'year_end': {
            'returns': year_end_returns,
            'mean': np.mean(year_end_returns) if year_end_returns else 0,
            'count': len(year_end_returns)
        },
        'golden_week': {
            'returns': golden_week_returns,
            'mean': np.mean(golden_week_returns) if golden_week_returns else 0,
            'count': len(golden_week_returns)
        }
    }
    
    return monthly_stats, special_periods

def create_japanese_analysis_chart(results_dict, ticker, monthly_stats, special_periods, output_dir):
    """Create Japan-specific analysis charts"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker}: Japanese Turn-of-Month Effect Analysis', fontsize=16, fontweight='bold')
    
    # Chart 1: Holding Period Comparison
    if results_dict:
        holding_periods = list(results_dict.keys())
        tom_means = [results_dict[p]['tom_mean'] * 100 for p in holding_periods]
        baseline_means = [results_dict[p]['baseline_mean'] * 100 for p in holding_periods]
        
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
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}%', ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Monthly patterns with Japanese fiscal year highlighted
    if monthly_stats:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        means = []
        for month in range(1, 13):
            if month in monthly_stats:
                means.append(monthly_stats[month]['mean'] * 100)
            else:
                means.append(0)
        
        # Color code: April (fiscal year end) and December (calendar year end) in different color
        colors = []
        for i, month in enumerate(range(1, 13)):
            if month in [3, 4]:  # March-April (Japanese fiscal year end/start)
                colors.append('gold')
            elif month == 12:  # December
                colors.append('orange')
            elif means[i] > 0:
                colors.append('green')
            else:
                colors.append('red')
        
        bars = ax2.bar(months, means, color=colors, alpha=0.7)
        ax2.set_title('Monthly Turn-of-Month Returns\n(Gold=Fiscal Year, Orange=Year-end)')
        ax2.set_ylabel('Mean Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            if mean != 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{mean:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Chart 3: Japanese special periods
    if special_periods:
        periods = ['Year-end\n(Dec+Apr)', 'Golden Week\n(May)', 'Other Months']
        period_means = [
            special_periods['year_end']['mean'] * 100,
            special_periods['golden_week']['mean'] * 100,
            0  # Placeholder for other months average
        ]
        
        # Calculate other months average
        if monthly_stats:
            other_months = []
            for month in [1, 2, 6, 7, 8, 9, 10, 11]:  # Exclude Dec, Apr, May
                if month in monthly_stats:
                    other_months.extend([monthly_stats[month]['mean']])
            if other_months:
                period_means[2] = np.mean(other_months) * 100
        
        colors_special = ['gold', 'lightblue', 'gray']
        bars3 = ax3.bar(periods, period_means, color=colors_special, alpha=0.7)
        ax3.set_title('Japanese Special Periods Effect')
        ax3.set_ylabel('Mean Return (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels and sample sizes
        for i, (bar, mean) in enumerate(zip(bars3, period_means)):
            if mean != 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{mean:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                # Add sample size
                if i == 0:
                    count = special_periods['year_end']['count']
                elif i == 1:
                    count = special_periods['golden_week']['count']
                else:
                    count = sum(monthly_stats.get(m, {'count': 0})['count'] 
                              for m in [1, 2, 6, 7, 8, 9, 10, 11])
                
                ax3.text(bar.get_x() + bar.get_width()/2., -0.02,
                        f'n={count}', ha='center', va='top', fontsize=8)
    
    # Chart 4: Best strategy cumulative performance
    if results_dict:
        best_period = max(results_dict.keys(), key=lambda k: results_dict[k]['tom_mean'])
        best_results = results_dict[best_period]
        
        tom_cumulative = (1 + best_results['tom_returns']).cumprod()
        baseline_cumulative = (1 + best_results['baseline_returns']).cumprod()
        
        ax4.plot(range(len(tom_cumulative)), tom_cumulative.values, 
                color='red', linewidth=2, label=f'Turn-of-Month ({best_period}d)', alpha=0.8)
        ax4.plot(range(len(baseline_cumulative)), baseline_cumulative.values,
                color='blue', linewidth=2, label=f'Random Baseline ({best_period}d)', alpha=0.8)
        ax4.set_title(f'Cumulative Performance - Best Strategy ({best_period}d holding)')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Add final performance text
        final_tom = tom_cumulative.iloc[-1] if len(tom_cumulative) > 0 else 1
        final_baseline = baseline_cumulative.iloc[-1] if len(baseline_cumulative) > 0 else 1
        ax4.text(0.02, 0.98, f'Final: TOM {final_tom:.3f}, Baseline {final_baseline:.3f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ticker}_japanese_turn_of_month.png", dpi=150, bbox_inches='tight')
    plt.close()

def format_japanese_results(results_dict, ticker, special_periods):
    """Format results for Japanese market analysis"""
    summary = f"\n🏯 {ticker} Japanese Turn-of-Month Analysis:\n"
    summary += "=" * 55 + "\n"
    
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
    
    # Add Japanese special periods
    summary += f"\n🎌 Japanese Special Periods:\n"
    summary += f"  Year-end (Dec+Apr): {special_periods['year_end']['mean']*100:.3f}% ({special_periods['year_end']['count']} trades)\n"
    summary += f"  Golden Week (May): {special_periods['golden_week']['mean']*100:.3f}% ({special_periods['golden_week']['count']} trades)\n"
    
    return summary

def main():
    print("🏯 Japanese Market Turn-of-Month Effect Analysis")
    print("=" * 60)
    print("Strategy: Buy last trading day of month, sell after N days")
    print("Focus: Japanese markets, indices, ETFs, and major stocks")
    print(f"Testing holding periods: {HOLDING_PERIODS} days")
    print()
    
    # Create output directory
    output_dir = "japanese_turn_of_month_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    summary_data = []
    
    # Test each Japanese market category
    for category, tickers in JAPANESE_TICKERS.items():
        print(f"\n🏯 Testing {category}:")
        print("-" * 40)
        
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}:")
            
            # Download data with Japanese market handling
            data = download_japanese_data(ticker)
            if data is None:
                continue
            
            # Analyze turn-of-month effect
            results = analyze_turn_of_month_by_holding_period(data, ticker)
            if not results:
                print(f"  ❌ Insufficient data for analysis")
                continue
            
            all_results[ticker] = results
            
            # Analyze Japanese-specific patterns
            monthly_stats, special_periods = analyze_japanese_market_patterns(data)
            
            # Print summary
            summary = format_japanese_results(results, ticker, special_periods)
            print(summary)
            
            # Create Japanese-specific charts
            create_japanese_analysis_chart(results, ticker, monthly_stats, special_periods, output_dir)
            
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
                    'P_Value': result['p_value'],
                    'Year_End_Effect': special_periods['year_end']['mean'],
                    'Golden_Week_Effect': special_periods['golden_week']['mean']
                })
    
    # Create summary analysis
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/japanese_turn_of_month_summary.csv", index=False)
        
        # Overall conclusion for Japanese markets
        print("\n" + "=" * 70)
        print("🎯 JAPANESE MARKET CONCLUSIONS")
        print("=" * 70)
        
        # Best performing strategies
        best_strategies = summary_df.nlargest(3, 'Difference')
        worst_strategies = summary_df.nsmallest(3, 'Difference')
        
        print("\n🏆 Best Japanese Turn-of-Month Strategies:")
        for _, row in best_strategies.iterrows():
            significance = "✅" if row['P_Value'] < 0.05 else "❌"
            print(f"  {row['Ticker']} ({row['Holding_Period']}d): +{row['Difference']*100:.3f}% {significance}")
        
        print("\n📉 Worst Japanese Turn-of-Month Strategies:")
        for _, row in worst_strategies.iterrows():
            significance = "✅" if row['P_Value'] < 0.05 else "❌"
            print(f"  {row['Ticker']} ({row['Holding_Period']}d): {row['Difference']*100:.3f}% {significance}")
        
        # Statistical significance summary
        significant_count = sum(1 for _, row in summary_df.iterrows() if row['P_Value'] < 0.05)
        total_tests = len(summary_df)
        
        print(f"\n📊 Statistical Significance in Japanese Markets:")
        print(f"  {significant_count}/{total_tests} strategies show significant turn-of-month effect")
        print(f"  {(significant_count/total_tests)*100:.1f}% of tests are statistically significant")
        
        # Average effect across Japanese strategies
        avg_difference = summary_df['Difference'].mean()
        print(f"\n📈 Average Japanese Turn-of-Month Effect: {avg_difference*100:.3f}% per trade")
        
        # Japanese special periods analysis
        avg_year_end = summary_df['Year_End_Effect'].mean()
        avg_golden_week = summary_df['Golden_Week_Effect'].mean()
        print(f"\n🎌 Japanese Special Period Effects:")
        print(f"  Year-end (Dec+Apr) average: {avg_year_end*100:.3f}%")
        print(f"  Golden Week (May) average: {avg_golden_week*100:.3f}%")
        
        if avg_difference > 0:
            print("\n✅ Overall positive turn-of-month effect in Japanese markets!")
        else:
            print("\n❌ No significant turn-of-month effect in Japanese markets overall")
        
        # Compare to fiscal year vs calendar year effects
        fiscal_sensitive = summary_df[summary_df['Year_End_Effect'] > 0.001].shape[0]
        print(f"\n🗾 Japanese Market Characteristics:")
        print(f"  {fiscal_sensitive}/{len(summary_df)} assets show positive year-end effects")
        print(f"  Suggests {(fiscal_sensitive/len(summary_df))*100:.1f}% may be sensitive to Japanese fiscal calendar")
    
    print(f"\n📁 Japanese analysis complete! Results saved to: {output_dir}/")
    print("Files created:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")

if __name__ == "__main__":
    main()