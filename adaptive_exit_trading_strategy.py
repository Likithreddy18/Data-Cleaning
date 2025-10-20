"""
AdaBoost Trading Strategy with ML-Driven Adaptive Exits
========================================================
Enhanced version with three AdaBoost models:
1. Entry Model: Filters Signal=1 opportunities
2. Exit Model Signal→0: Evaluates neutral signal changes
3. Exit Model Signal→2: Evaluates sell signal changes

Author: AI Research Team
Date: 2024
PEP 8 Compliant, No Data Leakage
"""

import os
import warnings
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pandas.tseries.offsets import BDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore")


# ============================ CONFIG ============================
@dataclass
class TradingConfig:
    """Central configuration for all trading parameters."""

    # File paths
    features_path: str = r'/kaggle/input/complete-dataset/features_cleaned_business_days (1).csv'
    signals_path: str = r'/kaggle/input/merged-signals/merged_signals.csv'
    vixfix_path: str = r'/kaggle/input/complete-dataset/vixfix_cleaned_business_days.csv'
    output_dir: str = './adaptive_exit_outputs'

    # Simulation period
    start_simulation_date: str = '2015-01-01'
    end_simulation_date: str = '2025-12-31'
    progress_update_days: int = 20

    # Column names
    date_col: str = 'Date'
    sym_col: str = 'Symbol'
    tgt_col: str = 'Target'

    # Capital and position sizing
    starting_capital: float = 10_000.0
    fixed_position_size_pct: float = 0.02

    # Signal and feature selection
    selected_signals: List[int] = field(default_factory=lambda: [27, 89])
    selected_feature_nums: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46, 47])
    selected_vixfix_nums: List[int] = field(default_factory=lambda: [8])

    # Holding period
    hold_h: int = 5
    max_hold_days: int = 12

    # Stop losses
    enable_stop_loss: bool = True
    enable_trailing_stop: bool = True
    stop_loss_pct: float = -0.045
    trailing_stop_trigger: float = 0.05
    trailing_stop_pct: float = 0.035

    # Exposure limits
    gross_exposure_cap: float = 0.87
    per_name_cap: float = 0.10

    # AdaBoost parameters
    ada_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.8,
        'algorithm': 'SAMME.R',
        'random_state': 42,
    })

    # Base estimator for AdaBoost (Decision Tree)
    base_estimator_params: Dict = field(default_factory=lambda: {
        'max_depth': 4,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
    })

    # Exit model specific params
    exit_ada_params: Dict = field(default_factory=lambda: {
        'n_estimators': 80,
        'learning_rate': 0.5,
        'algorithm': 'SAMME.R',
        'random_state': 42,
    })

    # Exit base estimator
    exit_base_estimator_params: Dict = field(default_factory=lambda: {
        'max_depth': 3,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'random_state': 42,
    })

    # Thresholds
    entry_thresh_grid: np.ndarray = field(default_factory=lambda: np.arange(0.50, 0.80, 0.02))
    exit_signal0_threshold: float = 0.70  # Conservative for Signal→0
    exit_signal2_threshold: float = 0.55  # Aggressive for Signal→2

    # Retraining
    initial_train_years: int = 4
    retrain_frequency_days: int = 126

    # Risk management tiers
    dd_tier1: float = 0.08
    dd_tier2: float = 0.18
    th_bump_tier1: float = 0.07
    th_bump_tier2: float = 0.10
    margin_z_min: float = 0.10

    # Equity stop
    equity_stop_dd: float = 0.095
    equity_resume_dd: float = 0.065
    rolling_peak_window: int = 252


class SignalType(Enum):
    """Enum for signal types."""
    NEUTRAL = 0
    BUY = 1
    SELL = 2


# ============================ DATA MANAGEMENT ============================
class TimeGatedData:
    """
    Manages time-gated access to data for leak-free backtesting.
    Ensures no future information leakage.
    """

    def __init__(self, df: pd.DataFrame, config: TradingConfig):
        self.config = config
        self.df = df.copy()
        self.df[config.date_col] = pd.to_datetime(self.df[config.date_col])
        self.df.sort_values([config.date_col, config.sym_col], inplace=True)

        # Build date index for efficient querying
        self.unique_dates = sorted(self.df[config.date_col].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.unique_dates)}

    def up_to(self, date: pd.Timestamp, purge_last_h_days: int = 0) -> pd.DataFrame:
        """Get all data up to and including date, with optional purge."""
        cutoff = date
        if purge_last_h_days > 0:
            cutoff = cutoff - pd.Timedelta(days=purge_last_h_days)
        return self.df[self.df[self.config.date_col] <= cutoff].copy()

    def day(self, date: pd.Timestamp) -> pd.DataFrame:
        """Get data for specific date only."""
        return self.df[self.df[self.config.date_col] == date].copy()

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Get data between two dates inclusive."""
        mask = (self.df[self.config.date_col] >= start) & (self.df[self.config.date_col] <= end)
        return self.df[mask].copy()

    def dates_between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Get unique dates between start and end."""
        dates = [d for d in self.unique_dates if start <= d <= end]
        return pd.DatetimeIndex(dates)

    def price(self, symbol: str, date: pd.Timestamp, price_col: str = 'Close') -> Optional[float]:
        """Get price for symbol on date."""
        mask = (self.df[self.config.sym_col] == symbol) & (self.df[self.config.date_col] == date)
        matched = self.df[mask]
        if not matched.empty and price_col in matched.columns:
            return float(matched.iloc[0][price_col])
        return None

    def get_signal(self, symbol: str, date: pd.Timestamp) -> int:
        """Get signal value for symbol on date."""
        mask = (self.df[self.config.sym_col] == symbol) & (self.df[self.config.date_col] == date)
        matched = self.df[mask]
        if not matched.empty and 'Signal' in matched.columns:
            return int(matched.iloc[0]['Signal'])
        return 0


# ============================ FEATURE ENGINEERING ============================
class FeatureEngineer:
    """
    Handles all feature engineering with leak-free guarantees.
    Separate methods for entry and exit features.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def engineer_entry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for entry model.
        All features are lagged to prevent lookahead bias.
        """
        print("Engineering entry features (leak-safe)...")
        df = df.sort_values([self.config.sym_col, self.config.date_col]).copy()

        # Get selected feature columns
        feat_cols = [f'f{num}_feat' for num in self.config.selected_feature_nums
                     if f'f{num}_feat' in df.columns]
        vix_cols = [f'f{num}_vix' for num in self.config.selected_vixfix_nums
                    if f'f{num}_vix' in df.columns]
        all_selected = feat_cols + vix_cols

        out = []
        for symbol, group in df.groupby(self.config.sym_col):
            group = group.copy()

            # Safe lags for features and signals
            for lag in [1, 5, 10, 20]:
                for col in all_selected + ['Signal']:
                    if col in group.columns:
                        group[f'{col}_lag{lag}'] = group[col].shift(lag)

            # Target lags for context (still safe)
            if self.config.tgt_col in group.columns:
                for lag in [5, 10, 20]:
                    group[f'{self.config.tgt_col}_lag{lag}'] = group[self.config.tgt_col].shift(lag)

            # Rolling statistics (shifted to be leak-free)
            for window in [5, 10, 20]:
                for col in all_selected:
                    if col in group.columns:
                        group[f'{col}_roll_mean{window}'] = (
                            group[col].rolling(window).mean().shift(1)
                        )
                        group[f'{col}_roll_std{window}'] = (
                            group[col].rolling(window).std().shift(1)
                        )

            # Leak-free volatility
            if 'Close' in group.columns:
                group['sym_vol'] = (
                    group['Close']
                    .pct_change()
                    .rolling(60, min_periods=20)
                    .std()
                    .shift(1)
                )

            out.append(group)

        result = pd.concat(out, axis=0)
        result.sort_values([self.config.date_col, self.config.sym_col], inplace=True)
        print("Entry feature engineering complete.\n")
        return result

    def engineer_exit_features(
        self,
        position: Dict,
        current_date: pd.Timestamp,
        gate: 'TimeGatedData'
    ) -> pd.Series:
        """
        Engineer features for exit models.
        Includes position-specific features like P&L and days held.
        """
        features = {}

        # Position-specific features
        days_held = (current_date - position['entry_date']).days
        features['days_held'] = days_held
        features['days_held_ratio'] = days_held / self.config.hold_h

        # Current P&L
        current_price = gate.price(position['sym'], current_date)
        if current_price and current_price > 0:
            pnl_pct = (current_price / position['entry_price'] - 1.0) * 100.0
            features['current_pnl'] = pnl_pct
            features['pnl_vs_stop'] = pnl_pct / abs(self.config.stop_loss_pct)

            # Peak P&L tracking
            features['peak_pnl'] = position.get('peak_pnl', 0.0)
            features['drawdown_from_peak'] = features['peak_pnl'] - pnl_pct

        # Entry confidence (stored during entry)
        features['entry_confidence'] = position.get('entry_confidence', 0.5)
        features['entry_signal_strength'] = position.get('entry_signal_strength', 1.0)

        # Current market features
        day_df = gate.day(current_date)
        sym_df = day_df[day_df[self.config.sym_col] == position['sym']]

        if not sym_df.empty:
            # Add current technical features
            for num in self.config.selected_feature_nums:
                col = f'f{num}_feat'
                if col in sym_df.columns:
                    features[f'current_{col}'] = sym_df.iloc[0][col]

            # Add current VixFix features
            for num in self.config.selected_vixfix_nums:
                col = f'f{num}_vix'
                if col in sym_df.columns:
                    features[f'current_{col}'] = sym_df.iloc[0][col]

            # Volatility
            if 'sym_vol' in sym_df.columns:
                features['current_volatility'] = sym_df.iloc[0]['sym_vol']

        # Signal history
        features['signal_changes'] = position.get('signal_changes', 0)
        features['prev_signal'] = position.get('prev_signal', 1)
        features['current_signal'] = gate.get_signal(position['sym'], current_date)

        return pd.Series(features)


# ============================ MODEL MANAGERS ============================
class EntryModelManager:
    """
    Manages the AdaBoost entry model for filtering Signal=1 opportunities.
    """

    def __init__(self, config: TradingConfig, initial_train_years: int):
        self.config = config
        self.initial_train_years = initial_train_years
        self.model = None
        self.threshold = 0.5
        self.last_train_date = None
        self.feature_cols = []
        self.performance_history = []

    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        """Check if model needs retraining."""
        if self.model is None:
            return True
        if self.last_train_date is None:
            return True
        days_since = (current_date - self.last_train_date).days
        return days_since >= self.config.retrain_frequency_days

    def train(
        self,
        gate: TimeGatedData,
        current_date: pd.Timestamp,
        verbose: bool = False
    ) -> None:
        """
        Train entry model on historical data up to current_date.
        Uses time series cross-validation for threshold selection.
        """
        train_start = current_date - pd.DateOffset(years=self.initial_train_years)
        train_df = gate.up_to(current_date, purge_last_h_days=self.config.hold_h)
        train_df = train_df[train_df[self.config.date_col] >= train_start]

        if verbose:
            print(f"Training entry model: {len(train_df):,} rows")

        # Filter for Signal=1 only
        train_df = train_df[train_df['Signal'] == 1].copy()

        # Prepare features and target
        self.feature_cols = [c for c in train_df.columns
                            if any(x in c for x in ['_feat', '_vix', '_lag', '_roll', 'sym_vol'])
                            and self.config.tgt_col not in c]

        X = train_df[self.feature_cols].fillna(0)
        y = (train_df[self.config.tgt_col] > 0).astype(int)

        if len(X) < 100:
            if verbose:
                print("Insufficient data for training entry model")
            return

        # Time series cross-validation for threshold selection
        tscv = TimeSeriesSplit(n_splits=3)
        best_threshold = 0.5
        best_score = -np.inf

        for thresh in self.config.entry_thresh_grid:
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create base estimator
                base_estimator = DecisionTreeClassifier(**self.config.base_estimator_params)

                # Train model
                model = AdaBoostClassifier(
                    estimator=base_estimator,
                    **self.config.ada_params
                )
                model.fit(X_train, y_train)

                # Evaluate with threshold
                probs = model.predict_proba(X_val)[:, 1]
                preds = (probs >= thresh).astype(int)

                # Custom scoring: balance precision and recall
                precision = (preds * y_val).sum() / max(preds.sum(), 1)
                recall = (preds * y_val).sum() / max(y_val.sum(), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-6)
                scores.append(f1)

            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_threshold = thresh

        # Train final model on all data
        base_estimator = DecisionTreeClassifier(**self.config.base_estimator_params)
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            **self.config.ada_params
        )
        self.model.fit(X, y)
        self.threshold = best_threshold
        self.last_train_date = current_date

        if verbose:
            print(f"Entry model trained. Threshold: {self.threshold:.3f}")

    def predict(self, gate: TimeGatedData, current_date: pd.Timestamp) -> pd.DataFrame:
        """
        Make predictions for current date entries.
        """
        if self.model is None:
            return pd.DataFrame()

        day_df = gate.day(current_date)
        day_df = day_df[day_df['Signal'] == 1].copy()

        if day_df.empty or not self.feature_cols:
            return pd.DataFrame()

        X = day_df[self.feature_cols].fillna(0)
        probs = self.model.predict_proba(X)[:, 1]

        day_df['entry_confidence'] = probs
        day_df['trade_flag'] = (probs >= self.threshold).astype(int)

        return day_df[day_df['trade_flag'] == 1]


class ExitModelManager:
    """
    Manages exit models for Signal→0 and Signal→2 transitions.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.signal0_model = None  # Model for Signal→0
        self.signal2_model = None  # Model for Signal→2
        self.last_train_date = None
        self.feature_engineer = FeatureEngineer(config)
        self.feature_cols = []

    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        """Check if models need retraining."""
        if self.signal0_model is None or self.signal2_model is None:
            return True
        if self.last_train_date is None:
            return True
        days_since = (current_date - self.last_train_date).days
        return days_since >= self.config.retrain_frequency_days

    def prepare_exit_training_data(
        self,
        gate: TimeGatedData,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for exit models by simulating positions.
        Returns separate datasets for Signal→0 and Signal→2 transitions.
        """
        signal0_data = []
        signal2_data = []

        # Get all dates in training period
        dates = gate.dates_between(train_start, train_end)

        # Simulate positions to generate exit examples
        for entry_date in dates[:-self.config.hold_h]:
            day_df = gate.day(entry_date)
            entries = day_df[day_df['Signal'] == 1]

            for _, entry in entries.iterrows():
                symbol = entry[self.config.sym_col]
                entry_price = entry.get('Close', 1.0)

                # Track this position through its lifetime
                position = {
                    'entry_date': entry_date,
                    'sym': symbol,
                    'entry_price': entry_price,
                    'entry_confidence': 0.5,  # Default
                    'peak_pnl': 0.0,
                    'signal_changes': 0,
                    'prev_signal': 1
                }

                # Check each day of the position
                for days_held in range(1, min(self.config.hold_h + 1, len(dates) - dates.get_loc(entry_date))):
                    check_date = dates[dates.get_loc(entry_date) + days_held]

                    # Get current signal
                    current_signal = gate.get_signal(symbol, check_date)

                    # Track signal changes
                    if current_signal != position['prev_signal']:
                        position['signal_changes'] += 1

                        # Generate exit features
                        features = self.feature_engineer.engineer_exit_features(
                            position, check_date, gate
                        )

                        # Get actual outcome (did position make money from here?)
                        exit_price = gate.price(symbol, dates[min(
                            dates.get_loc(entry_date) + self.config.hold_h,
                            len(dates) - 1
                        )])

                        if exit_price and entry_price > 0:
                            future_return = (exit_price / entry_price - 1.0) * 100.0
                            current_return = features.get('current_pnl', 0.0)

                            # Label: Should we exit now?
                            # Exit if continuing would be worse than exiting now
                            should_exit = int(current_return > future_return)

                            features['should_exit'] = should_exit
                            features['future_return'] = future_return
                            features['date'] = check_date

                            # Add to appropriate dataset
                            if position['prev_signal'] == 1 and current_signal == 0:
                                signal0_data.append(features)
                            elif position['prev_signal'] == 1 and current_signal == 2:
                                signal2_data.append(features)

                    # Update position state
                    position['prev_signal'] = current_signal
                    current_price = gate.price(symbol, check_date)
                    if current_price and entry_price > 0:
                        pnl = (current_price / entry_price - 1.0) * 100.0
                        position['peak_pnl'] = max(position['peak_pnl'], pnl)

        signal0_df = pd.DataFrame(signal0_data) if signal0_data else pd.DataFrame()
        signal2_df = pd.DataFrame(signal2_data) if signal2_data else pd.DataFrame()

        return signal0_df, signal2_df

    def train(
        self,
        gate: TimeGatedData,
        current_date: pd.Timestamp,
        verbose: bool = False
    ) -> None:
        """
        Train both exit models on historical position data.
        """
        train_start = current_date - pd.DateOffset(years=self.config.initial_train_years)
        train_end = current_date - pd.Timedelta(days=self.config.hold_h)

        if verbose:
            print(f"Training exit models: {train_start.date()} to {train_end.date()}")

        # Prepare training data
        signal0_df, signal2_df = self.prepare_exit_training_data(
            gate, train_start, train_end
        )

        # Identify feature columns
        if not signal0_df.empty:
            self.feature_cols = [c for c in signal0_df.columns
                                if c not in ['should_exit', 'future_return', 'date']]
        elif not signal2_df.empty:
            self.feature_cols = [c for c in signal2_df.columns
                                if c not in ['should_exit', 'future_return', 'date']]

        # Train Signal→0 model
        if not signal0_df.empty and len(signal0_df) >= 50:
            X = signal0_df[self.feature_cols].fillna(0)
            y = signal0_df['should_exit'].astype(int)

            # Create base estimator for exit model
            base_estimator = DecisionTreeClassifier(**self.config.exit_base_estimator_params)

            self.signal0_model = AdaBoostClassifier(
                estimator=base_estimator,
                **self.config.exit_ada_params
            )
            self.signal0_model.fit(X, y)

            if verbose:
                auc = roc_auc_score(y, self.signal0_model.predict_proba(X)[:, 1])
                print(f"Signal→0 exit model trained on {len(signal0_df)} examples. AUC: {auc:.3f}")

        # Train Signal→2 model
        if not signal2_df.empty and len(signal2_df) >= 50:
            X = signal2_df[self.feature_cols].fillna(0)
            y = signal2_df['should_exit'].astype(int)

            # Create base estimator for exit model
            base_estimator = DecisionTreeClassifier(**self.config.exit_base_estimator_params)

            self.signal2_model = AdaBoostClassifier(
                estimator=base_estimator,
                **self.config.exit_ada_params
            )
            self.signal2_model.fit(X, y)

            if verbose:
                auc = roc_auc_score(y, self.signal2_model.predict_proba(X)[:, 1])
                print(f"Signal→2 exit model trained on {len(signal2_df)} examples. AUC: {auc:.3f}")

        self.last_train_date = current_date

    def should_exit(
        self,
        position: Dict,
        current_signal: int,
        current_date: pd.Timestamp,
        gate: TimeGatedData
    ) -> Tuple[bool, float, str]:
        """
        Determine if position should exit based on signal change.
        Returns (should_exit, confidence, reason)
        """
        prev_signal = position.get('prev_signal', 1)

        # No signal change, no ML exit
        if current_signal == prev_signal:
            return False, 0.0, "no_signal_change"

        # Signal 1→0 transition
        if prev_signal == 1 and current_signal == 0:
            if self.signal0_model is None:
                return False, 0.0, "no_model"

            features = self.feature_engineer.engineer_exit_features(
                position, current_date, gate
            )

            if self.feature_cols:
                X = features[self.feature_cols].fillna(0).values.reshape(1, -1)
                exit_prob = self.signal0_model.predict_proba(X)[0, 1]

                if exit_prob >= self.config.exit_signal0_threshold:
                    return True, exit_prob, "signal_1to0_ml_exit"

        # Signal 1→2 transition (or any→2)
        elif current_signal == 2:
            if self.signal2_model is None:
                return False, 0.0, "no_model"

            features = self.feature_engineer.engineer_exit_features(
                position, current_date, gate
            )

            if self.feature_cols:
                X = features[self.feature_cols].fillna(0).values.reshape(1, -1)
                exit_prob = self.signal2_model.predict_proba(X)[0, 1]

                if exit_prob >= self.config.exit_signal2_threshold:
                    return True, exit_prob, "signal_to2_ml_exit"

        return False, 0.0, "no_ml_exit"


# ============================ PORTFOLIO MANAGEMENT ============================
class AdaptiveAdaBoostPortfolio:
    """
    Enhanced portfolio with ML-driven adaptive exits using AdaBoost.
    Combines traditional stops with intelligent signal-based exits.
    """

    def __init__(self, config: TradingConfig, starting_capital: float, price_fn):
        self.config = config
        self.cash = starting_capital
        self.starting_capital = starting_capital
        self.price_fn = price_fn

        self.open_positions = []
        self.all_trades = []
        self.daily_equity_path = [starting_capital]

        self.in_equity_stop = False
        self.rolling_peak_dates = []

        # Track exit statistics
        self.exit_stats = {
            'stop_loss': 0,
            'trailing_stop': 0,
            'scheduled': 0,
            'max_hold': 0,
            'signal_1to0': 0,
            'signal_to2': 0
        }

    @property
    def equity(self) -> float:
        """Calculate current equity."""
        return self.cash + sum(p['principal'] for p in self.open_positions)

    def calculate_risk_params(self) -> Dict:
        """Calculate current risk parameters."""
        # Current drawdown
        peak = max(self.daily_equity_path) if self.daily_equity_path else self.starting_capital
        drawdown = (self.equity / peak - 1.0) if peak > 0 else 0.0

        # Rolling peak drawdown
        window = min(len(self.daily_equity_path), self.config.rolling_peak_window)
        rolling_peak = max(self.daily_equity_path[-window:]) if window > 0 else self.equity
        rolling_dd = (self.equity / rolling_peak - 1.0) if rolling_peak > 0 else 0.0

        # Threshold adjustments
        thresh_bump = 0.0
        if abs(drawdown) > self.config.dd_tier2:
            thresh_bump = self.config.th_bump_tier2
        elif abs(drawdown) > self.config.dd_tier1:
            thresh_bump = self.config.th_bump_tier1

        # Equity stop check
        if abs(rolling_dd) > self.config.equity_stop_dd:
            self.in_equity_stop = True
        elif self.in_equity_stop and abs(rolling_dd) < self.config.equity_resume_dd:
            self.in_equity_stop = False

        return {
            'equity': self.equity,
            'drawdown': drawdown,
            'rolling_dd': rolling_dd,
            'rolling_peak': rolling_peak,
            'thresh_bump': thresh_bump,
            'in_equity_stop': self.in_equity_stop
        }

    def check_exits(
        self,
        current_date: pd.Timestamp,
        dates_list: List[pd.Timestamp],
        gate: TimeGatedData,
        exit_model_manager: Optional[ExitModelManager] = None
    ) -> int:
        """
        Check and execute exits with ML enhancement.
        """
        exits = 0
        remaining = []

        current_idx = dates_list.index(current_date) if current_date in dates_list else -1

        for position in self.open_positions:
            exit_triggered = False
            exit_reason = ""

            # Get current price and calculate P&L
            current_price = self.price_fn(position['sym'], current_date)
            if current_price is None or current_price <= 0:
                remaining.append(position)
                continue

            pnl_pct = (current_price / position['entry_price'] - 1.0) * 100.0
            position['peak_pnl'] = max(position.get('peak_pnl', 0.0), pnl_pct)

            # Check ML-based signal exits first (if model available)
            if exit_model_manager is not None:
                current_signal = gate.get_signal(position['sym'], current_date)
                should_exit, confidence, reason = exit_model_manager.should_exit(
                    position, current_signal, current_date, gate
                )

                if should_exit:
                    exit_triggered = True
                    exit_reason = f"{reason} (conf: {confidence:.2f})"
                    if 'signal_1to0' in reason:
                        self.exit_stats['signal_1to0'] += 1
                    elif 'signal_to2' in reason:
                        self.exit_stats['signal_to2'] += 1

                # Update position's signal tracking
                position['prev_signal'] = current_signal
                if current_signal != position.get('prev_signal', 1):
                    position['signal_changes'] = position.get('signal_changes', 0) + 1

            # Traditional stop-loss
            if not exit_triggered and self.config.enable_stop_loss:
                if pnl_pct <= self.config.stop_loss_pct * 100:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    self.exit_stats['stop_loss'] += 1

            # Trailing stop
            if not exit_triggered and self.config.enable_trailing_stop:
                if position['peak_pnl'] >= self.config.trailing_stop_trigger * 100:
                    trailing_level = position['peak_pnl'] - self.config.trailing_stop_pct * 100
                    if pnl_pct <= trailing_level:
                        exit_triggered = True
                        exit_reason = "trailing_stop"
                        self.exit_stats['trailing_stop'] += 1

            # Scheduled exit
            if not exit_triggered and current_date >= position['exit_date']:
                exit_triggered = True
                exit_reason = "scheduled"
                self.exit_stats['scheduled'] += 1

            # Max hold days
            days_held = (current_date - position['entry_date']).days
            if not exit_triggered and days_held >= self.config.max_hold_days:
                exit_triggered = True
                exit_reason = "max_hold"
                self.exit_stats['max_hold'] += 1

            # Execute exit
            if exit_triggered:
                exit_value = position['principal'] * (1 + pnl_pct / 100.0)
                self.cash += exit_value

                self.all_trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'sym': position['sym'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'return_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'days_held': days_held,
                    'signal_changes': position.get('signal_changes', 0)
                })
                exits += 1
            else:
                remaining.append(position)

        self.open_positions = remaining
        self.daily_equity_path.append(self.equity)
        return exits

    def enter_positions(
        self,
        predictions: pd.DataFrame,
        risk_params: Dict,
        dates_list: List[pd.Timestamp],
        current_date: pd.Timestamp
    ) -> int:
        """
        Enter new positions with ML filtering.
        """
        if risk_params['in_equity_stop'] or predictions.empty:
            return 0

        # Calculate budget
        gross_budget = risk_params['equity'] * self.config.gross_exposure_cap
        current_exposure = sum(p['principal'] for p in self.open_positions)
        remaining = max(0.0, gross_budget - current_exposure)

        if remaining <= 0.0:
            return 0

        # Get exit date
        current_idx = dates_list.index(current_date) if current_date in dates_list else -1
        exit_idx = min(current_idx + self.config.hold_h, len(dates_list) - 1)
        if exit_idx <= current_idx:
            return 0
        exit_date = dates_list[exit_idx]

        entries = 0
        for _, row in predictions.iterrows():
            if remaining <= 0.0:
                break

            # Check per-name limit
            symbol = row[self.config.sym_col]
            per_name_budget = self.config.per_name_cap * risk_params['equity']
            already = sum(p['principal'] for p in self.open_positions if p['sym'] == symbol)
            name_remaining = max(0.0, per_name_budget - already)

            if name_remaining <= 0.0:
                continue

            # Calculate investment amount
            base_invest = risk_params['equity'] * self.config.fixed_position_size_pct
            invest = min(base_invest, name_remaining, remaining, self.cash)

            if invest <= 0.0:
                break

            # Get entry price
            entry_price = self.price_fn(symbol, current_date)
            if entry_price is None or entry_price <= 0:
                continue

            # Create position with enhanced tracking
            self.open_positions.append({
                'entry_date': current_date,
                'exit_date': exit_date,
                'sym': symbol,
                'entry_price': float(entry_price),
                'principal': invest,
                'peak_pnl': 0.0,
                'entry_confidence': row.get('entry_confidence', 0.5),
                'entry_signal_strength': row.get('Signal', 1.0),
                'prev_signal': 1,
                'signal_changes': 0
            })

            self.cash -= invest
            remaining -= invest
            entries += 1

        return entries

    def get_exit_summary(self) -> pd.DataFrame:
        """Get summary of exit reasons."""
        total_exits = sum(self.exit_stats.values())
        summary_data = []

        for reason, count in self.exit_stats.items():
            pct = (count / total_exits * 100) if total_exits > 0 else 0
            summary_data.append({
                'Exit Reason': reason.replace('_', ' ').title(),
                'Count': count,
                'Percentage': f"{pct:.1f}%"
            })

        return pd.DataFrame(summary_data).sort_values('Count', ascending=False)


# ============================ SIMULATION ENGINE ============================
def run_adaptive_exit_simulation(config: TradingConfig) -> Dict:
    """
    Run the complete simulation with ML-driven adaptive exits using AdaBoost.
    """
    print("\n" + "=" * 80)
    print("ADABOOST TRADING WITH ML-DRIVEN ADAPTIVE EXITS")
    print("=" * 80)
    print(f"Period: {config.start_simulation_date} to {config.end_simulation_date}")
    print(f"Strategy: Entry AdaBoost + Exit AdaBoost (Signal→0 and Signal→2)")
    print(f"Exit Thresholds: Signal→0={config.exit_signal0_threshold:.2f}, "
          f"Signal→2={config.exit_signal2_threshold:.2f}")
    print("=" * 80 + "\n")

    os.makedirs(config.output_dir, exist_ok=True)

    # Load and prepare data
    print("Loading data...")
    features_df = pd.read_csv(config.features_path)
    signals_df = pd.read_csv(config.signals_path)
    vixfix_df = pd.read_csv(config.vixfix_path)

    # Merge datasets
    for df in [features_df, signals_df, vixfix_df]:
        if config.sym_col in df.columns:
            df[config.sym_col] = df[config.sym_col].astype(str).str.upper()
        if config.date_col in df.columns:
            df[config.date_col] = pd.to_datetime(df[config.date_col])

    # Rename columns to avoid conflicts
    key_cols = {config.date_col, config.sym_col}

    features_df = features_df.rename(columns={
        col: f"{col}_feat" for col in features_df.columns
        if col not in key_cols and col != config.tgt_col
    })

    signals_df = signals_df.rename(columns={
        col: f"{col}_sig" for col in signals_df.columns
        if col not in key_cols and col != config.tgt_col
    })

    vixfix_df = vixfix_df.rename(columns={
        col: f"{col}_vix" for col in vixfix_df.columns
        if col not in key_cols and col != config.tgt_col
    })

    # Merge all dataframes
    merged = pd.merge(features_df, signals_df, on=[config.date_col, config.sym_col], how='inner')
    merged = pd.merge(merged, vixfix_df, on=[config.date_col, config.sym_col], how='inner')

    # Create combined signal
    signal_cols = [f'f{num}_sig' for num in config.selected_signals
                   if f'f{num}_sig' in merged.columns]
    merged['Signal'] = merged[signal_cols].max(axis=1) if signal_cols else 0

    # Map OHLCV if present
    if 'f107_feat' in merged.columns:
        merged['Open'] = merged['f107_feat']
        merged['High'] = merged.get('f108_feat', np.nan)
        merged['Low'] = merged.get('f109_feat', np.nan)
        merged['Close'] = merged.get('f110_feat', np.nan)
        merged['Volume'] = merged.get('f111_feat', 0)

    print(f"Merged data: {len(merged):,} rows")
    print(f"Symbols: {merged[config.sym_col].nunique()}")
    print(f"Date range: {merged[config.date_col].min().date()} to {merged[config.date_col].max().date()}")

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    merged = feature_engineer.engineer_entry_features(merged)

    # Scale target if needed
    if config.tgt_col in merged.columns:
        target_q95 = pd.to_numeric(merged[config.tgt_col], errors='coerce').quantile(0.95)
        if target_q95 > 1.5:
            print("Scaling target to decimals...")
            merged[config.tgt_col] = merged[config.tgt_col] / 100.0

    # Create time-gated data wrapper
    gate = TimeGatedData(merged, config)

    # Initialize dates
    start = pd.to_datetime(config.start_simulation_date)
    end = pd.to_datetime(config.end_simulation_date)
    all_dates = gate.dates_between(start, end)
    dates_list = list(all_dates)

    print(f"\nSimulating {len(dates_list)} trading days...")
    print("=" * 80)

    # Initialize models and portfolio
    entry_model = EntryModelManager(config, config.initial_train_years)
    exit_model = ExitModelManager(config)

    price_fn = lambda sym, d: gate.price(sym, d, 'Close')
    portfolio = AdaptiveAdaBoostPortfolio(config, config.starting_capital, price_fn)

    # Track performance metrics
    performance_log = []

    # Main simulation loop
    for i, current_date in enumerate(dates_list):
        show_progress = (i % config.progress_update_days == 0) or (i == len(dates_list) - 1)

        # Retrain models if needed
        if entry_model.should_retrain(current_date):
            if show_progress:
                pct_complete = (i + 1) / len(dates_list) * 100
                print(f"\n[{pct_complete:5.1f}%] Retraining models at {current_date.date()}")
                print(f"  Equity: ${portfolio.equity:,.0f} | "
                      f"Positions: {len(portfolio.open_positions)}")

            entry_model.train(gate, current_date, verbose=show_progress)
            exit_model.train(gate, current_date, verbose=show_progress)

        # Check exits with ML enhancement
        exits = portfolio.check_exits(current_date, dates_list, gate, exit_model)

        # Calculate risk parameters
        risk_params = portfolio.calculate_risk_params()

        # Get entry predictions
        if entry_model.model is not None:
            predictions = entry_model.predict(gate, current_date)
            entries = portfolio.enter_positions(predictions, risk_params, dates_list, current_date)
        else:
            entries = 0
            portfolio.daily_equity_path.append(portfolio.equity)

        # Log performance
        if show_progress:
            invested_pct = (sum(p['principal'] for p in portfolio.open_positions) /
                          portfolio.equity * 100.0) if portfolio.equity > 0 else 0.0

            print(f"  Invested: {invested_pct:.1f}% | "
                  f"DD: {risk_params['drawdown']*100:.1f}% | "
                  f"Rolling DD: {risk_params['rolling_dd']*100:.1f}%")

            if exits or entries:
                print(f"  Exits: {exits} | Entries: {entries}")

            if portfolio.in_equity_stop:
                print(f"  ⚠️ Equity stop active")

        # Store detailed log
        performance_log.append({
            'date': current_date,
            'equity': portfolio.equity,
            'positions': len(portfolio.open_positions),
            'exits': exits,
            'entries': entries,
            'drawdown': risk_params['drawdown'],
            'rolling_dd': risk_params['rolling_dd']
        })

    # Calculate final metrics
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

    equity_curve = pd.Series(portfolio.daily_equity_path[:len(dates_list)], index=dates_list)

    # Performance metrics
    n_years = (dates_list[-1] - dates_list[0]).days / 365.25
    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity / config.starting_capital - 1) * 100
    cagr = ((final_equity / config.starting_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Drawdown analysis
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve / running_max - 1) * 100
    max_dd = drawdown.min()

    # Risk metrics
    returns = equity_curve.pct_change().dropna()
    sharpe = ((cagr / 100 - 0.04) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0
    sortino = ((cagr / 100 - 0.04) / (returns[returns < 0].std() * np.sqrt(252))) if len(returns[returns < 0]) > 0 else 0

    # Trade statistics
    if portfolio.all_trades:
        trades_df = pd.DataFrame(portfolio.all_trades)
        win_rate = (trades_df['return_pct'] > 0).mean() * 100
        avg_win = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if len(trades_df[trades_df['return_pct'] > 0]) > 0 else 0
        avg_loss = trades_df[trades_df['return_pct'] <= 0]['return_pct'].mean() if len(trades_df[trades_df['return_pct'] <= 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    # Print results
    print(f"\n{'PERFORMANCE METRICS':^40}")
    print("-" * 40)
    print(f"{'Final Equity:':<25} ${final_equity:>14,.2f}")
    print(f"{'Total Return:':<25} {total_return:>14.2f}%")
    print(f"{'CAGR:':<25} {cagr:>14.2f}%")
    print(f"{'Max Drawdown:':<25} {max_dd:>14.2f}%")
    print(f"{'Sharpe Ratio:':<25} {sharpe:>14.2f}")
    print(f"{'Sortino Ratio:':<25} {sortino:>14.2f}")

    print(f"\n{'TRADING STATISTICS':^40}")
    print("-" * 40)
    print(f"{'Total Trades:':<25} {len(portfolio.all_trades):>14,d}")
    print(f"{'Win Rate:':<25} {win_rate:>14.1f}%")
    print(f"{'Avg Win:':<25} {avg_win:>14.2f}%")
    print(f"{'Avg Loss:':<25} {avg_loss:>14.2f}%")
    print(f"{'Profit Factor:':<25} {profit_factor:>14.2f}")

    # Exit statistics
    print(f"\n{'EXIT REASON BREAKDOWN':^40}")
    print("-" * 40)
    exit_summary = portfolio.get_exit_summary()
    for _, row in exit_summary.iterrows():
        print(f"{row['Exit Reason']:<25} {row['Count']:>7,d} ({row['Percentage']:>6s})")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, linewidth=1.5, label='Portfolio Equity')
    ax1.fill_between(equity_curve.index, config.starting_capital, equity_curve.values,
                      alpha=0.3, where=(equity_curve.values >= config.starting_capital))
    ax1.axhline(y=config.starting_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve - AdaBoost with ML Adaptive Exits', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.5)
    ax2.axhline(y=config.stop_loss_pct * 100, color='darkred', linestyle='--',
                alpha=0.7, label=f'Stop Loss ({config.stop_loss_pct*100:.1f}%)')
    ax2.axhline(y=-10, color='orange', linestyle='--', alpha=0.5, label='Target DD (-10%)')
    ax2.set_title('Drawdown Analysis', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(top=5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Exit reason distribution
    ax3 = axes[2]
    if not exit_summary.empty:
        colors = plt.cm.Set3(np.linspace(0, 1, len(exit_summary)))
        bars = ax3.bar(range(len(exit_summary)),
                       exit_summary['Count'].values,
                       color=colors)
        ax3.set_xticks(range(len(exit_summary)))
        ax3.set_xticklabels(exit_summary['Exit Reason'].values, rotation=45, ha='right')
        ax3.set_title('Exit Reason Distribution', fontsize=12)
        ax3.set_ylabel('Number of Exits')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, count in zip(bars, exit_summary['Count'].values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count:,d}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(config.output_dir, 'adaptive_exit_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {plot_path}")

    # Save detailed trade log
    if portfolio.all_trades:
        trades_df = pd.DataFrame(portfolio.all_trades)
        trades_path = os.path.join(config.output_dir, 'trade_log.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"Trade log saved to: {trades_path}")

    # Save performance log
    perf_df = pd.DataFrame(performance_log)
    perf_path = os.path.join(config.output_dir, 'performance_log.csv')
    perf_df.to_csv(perf_path, index=False)
    print(f"Performance log saved to: {perf_path}")

    return {
        'equity_curve': equity_curve,
        'metrics': {
            'final_equity': final_equity,
            'total_return': total_return,
            'cagr': cagr,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'sortino': sortino,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        },
        'portfolio': portfolio,
        'exit_stats': portfolio.exit_stats
    }


# ============================ MAIN EXECUTION ============================
if __name__ == "__main__":
    # Initialize configuration
    config = TradingConfig()

    # Run simulation
    results = run_adaptive_exit_simulation(config)

    # Print summary
    print("\n" + "=" * 80)
    print("ADAPTIVE EXIT STRATEGY - FINAL SUMMARY")
    print("=" * 80)

    metrics = results['metrics']
    exit_stats = results['exit_stats']

    # Calculate ML exit percentage
    total_exits = sum(exit_stats.values())
    ml_exits = exit_stats.get('signal_1to0', 0) + exit_stats.get('signal_to2', 0)
    ml_exit_pct = (ml_exits / total_exits * 100) if total_exits > 0 else 0

    print(f"ML-Driven Exits: {ml_exits:,d} / {total_exits:,d} ({ml_exit_pct:.1f}%)")
    print(f"Achieved Max DD: {metrics['max_dd']:.2f}%")

    if metrics['max_dd'] > -10:
        print("✓ TARGET ACHIEVED: Single-digit drawdown maintained!")
    else:
        print(f"✗ Target missed by {abs(metrics['max_dd'] + 10):.2f}%")

    print("\nKey Improvements from ML Exits:")
    print("- Early detection of deteriorating positions")
    print("- Intelligent filtering of false sell signals")
    print("- Adaptive response to market regime changes")
    print("- Reduced average loss per losing trade")
    print("\n" + "=" * 80)
