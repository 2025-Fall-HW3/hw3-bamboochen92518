"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=15, gamma=1):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def enhanced_mv_opt(self, R_n, gamma):
        """
        Technical analysis based momentum strategy
        """
        n = len(R_n.columns)
        
        # Calculate technical indicators
        prices = self.price[R_n.columns].iloc[-30:]  # Last 30 days of prices
        
        # 1. Moving average crossover signals
        ma_short = prices.rolling(5).mean().iloc[-1]
        ma_long = prices.rolling(20).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        # Trend strength: how far above long MA
        trend_strength = (current_price - ma_long) / ma_long
        
        # 2. Momentum signals
        returns_5d = R_n.iloc[-5:].mean()
        returns_20d = R_n.iloc[-20:].mean() if len(R_n) >= 20 else R_n.mean()
        
        # 3. Volatility-adjusted momentum
        volatility = R_n.std()
        risk_adjusted_momentum = returns_5d / (volatility + 1e-6)
        
        # 4. Combine signals
        # Bullish signal: price > MA_short > MA_long and positive momentum
        bullish_score = np.zeros(n)
        for i, asset in enumerate(R_n.columns):
            score = 0
            
            # Trend following component
            if current_price[asset] > ma_short[asset] > ma_long[asset]:
                score += 2.0
            elif current_price[asset] > ma_long[asset]:
                score += 1.0
                
            # Momentum component
            if returns_5d[asset] > 0:
                score += 1.0
            if returns_20d[asset] > 0:
                score += 0.5
                
            # Risk-adjusted momentum
            score += max(0, risk_adjusted_momentum[asset]) * 2
            
            # Trend strength bonus
            if trend_strength[asset] > 0.05:  # 5% above MA
                score += 1.0
                
            bullish_score[i] = score
        
        # Select top 3 assets with highest bullish scores
        top_n = min(3, max(2, n//4))
        top_indices = np.argsort(bullish_score)[-top_n:]
        
        # Aggressive concentration on winners
        weights = np.zeros(n)
        
        if len(top_indices) > 0 and np.sum(bullish_score[top_indices]) > 0:
            # Allocate based on bullish scores with concentration
            scores = bullish_score[top_indices]
            
            # Apply exponential weighting to amplify differences
            exp_scores = np.exp(scores / np.max(scores)) if np.max(scores) > 0 else np.ones_like(scores)
            normalized_weights = exp_scores / np.sum(exp_scores)
            
            # Set minimum allocation and concentrate on best performer
            max_weight = 0.6  # Allow up to 60% in single asset
            min_weight = 0.15  # Minimum 15% per selected asset
            
            # Ensure we don't exceed limits
            for i, idx in enumerate(top_indices):
                weight = max(min_weight, min(max_weight, normalized_weights[i]))
                weights[idx] = weight
            
            # Renormalize
            total = np.sum(weights)
            if total > 0:
                weights = weights / total
            else:
                weights[top_indices] = 1.0 / len(top_indices)
        else:
            # Fallback: equal weight among all assets
            weights[:] = 1.0 / n
            
        return weights.tolist()

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Enhanced Mean-Variance optimization with momentum and risk adjustment
        for i in range(self.lookback + 1, len(self.price)):
            # Get historical returns for the lookback period
            R_n = self.returns[assets].iloc[i - self.lookback : i]
            
            # Calculate optimal weights using enhanced mean-variance optimization
            weights = self.enhanced_mv_opt(R_n, self.gamma)
            
            # Assign weights for this time period
            for j, asset in enumerate(assets):
                self.portfolio_weights.loc[self.price.index[i], asset] = weights[j]
            
            # Set weight to 0 for excluded asset (SPY)
            self.portfolio_weights.loc[self.price.index[i], self.exclude] = 0.0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
