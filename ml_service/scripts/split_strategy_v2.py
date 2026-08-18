#!/usr/bin/env python3
"""
split_strategy_v2.py — GoMata Leakage Audit
Phase 2: True Generalization Splits

Implements 3 split strategies that prevent leakage:
  A) Animal-level: 80% animals train / 20% unseen test
  B) Time-based: first 70% timestamps / last 30%
  C) Combined: unseen animal × future time segment

Usage:
  from split_strategy_v2 import animal_time_split
  X_train, X_test, y_train, y_test = animal_time_split(df, features, label)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("SplitStrategy")


def animal_split(df, features, label_col, train_ratio=0.8, seed=42):
    """
    A) Animal-level split — NO animal overlap between train and test.
    """
    rng = np.random.default_rng(seed)
    animals = df['animal_id'].unique()
    rng.shuffle(animals)

    n_train = int(len(animals) * train_ratio)
    train_animals = set(animals[:n_train])
    test_animals = set(animals[n_train:])

    train_mask = df['animal_id'].isin(train_animals)
    test_mask = df['animal_id'].isin(test_animals)

    X_train = df.loc[train_mask, features].fillna(0)
    X_test = df.loc[test_mask, features].fillna(0)
    y_train = df.loc[train_mask, label_col]
    y_test = df.loc[test_mask, label_col]

    logger.info(f"Animal split: {len(train_animals)} train animals, {len(test_animals)} test animals")
    logger.info(f"  Train: {len(X_train)} rows ({y_train.sum():.0f} positive)")
    logger.info(f"  Test:  {len(X_test)} rows ({y_test.sum():.0f} positive)")

    return X_train, X_test, y_train, y_test


def time_split(df, features, label_col, train_ratio=0.7):
    """
    B) Time-based split — train on first 70%, test on last 30%.
    Applied per-animal to prevent temporal leakage.
    """
    train_rows, test_rows = [], []

    for animal_id, group in df.groupby('animal_id'):
        g = group.sort_index()  # already sorted by time
        n = len(g)
        split_idx = int(n * train_ratio)
        train_rows.append(g.iloc[:split_idx])
        test_rows.append(g.iloc[split_idx:])

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    logger.info(f"Time split: 70/30 per animal")
    logger.info(f"  Train: {len(X_train)} rows ({y_train.sum():.0f} positive)")
    logger.info(f"  Test:  {len(X_test)} rows ({y_test.sum():.0f} positive)")

    return X_train, X_test, y_train, y_test


def animal_time_split(df, features, label_col, animal_train_ratio=0.8,
                       time_train_ratio=0.7, seed=42):
    """
    C) Combined split — unseen animals × future time.
    
    Train set: 80% of animals, first 70% of their timestamps
    Test set:  20% unseen animals, last 30% of remaining animals
    
    This is the strictest anti-leakage split.
    """
    rng = np.random.default_rng(seed)
    animals = df['animal_id'].unique()
    rng.shuffle(animals)

    n_animal_train = int(len(animals) * animal_train_ratio)
    train_animals = animals[:n_animal_train]
    test_animals = animals[n_animal_train:]

    # Train: first 70% timestamps of TRAIN animals only
    train_rows = []
    for animal_id in train_animals:
        group = df[df['animal_id'] == animal_id]
        n = len(group)
        split_idx = int(n * time_train_ratio)
        train_rows.append(group.iloc[:split_idx])

    # Test set 1: Last 30% of train animals (temporal test)
    temporal_test_rows = []
    for animal_id in train_animals:
        group = df[df['animal_id'] == animal_id]
        n = len(group)
        split_idx = int(n * time_train_ratio)
        temporal_test_rows.append(group.iloc[split_idx:])

    # Test set 2: ALL data from unseen animals (animal test)
    unseen_test = df[df['animal_id'].isin(test_animals)]

    # Combined: merge both test sets
    train_df = pd.concat(train_rows)
    temporal_test_df = pd.concat(temporal_test_rows)
    combined_test_df = pd.concat([temporal_test_df, unseen_test])

    X_train = train_df[features].fillna(0)
    X_test = combined_test_df[features].fillna(0)
    y_train = train_df[label_col]
    y_test = combined_test_df[label_col]

    logger.info(f"Combined animal+time split:")
    logger.info(f"  Train animals: {len(train_animals)}, Test animals: {len(test_animals)}")
    logger.info(f"  Train: {len(X_train)} rows ({y_train.sum():.0f} positive, "
                f"{y_train.mean()*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} rows ({y_test.sum():.0f} positive, "
                f"{y_test.mean()*100:.1f}%)")
    logger.info(f"    ├ Temporal test (known animals, future time): {len(temporal_test_df)}")
    logger.info(f"    └ Unseen animal test: {len(unseen_test)}")

    return X_train, X_test, y_train, y_test, {
        "train_animals": len(train_animals),
        "test_animals": len(test_animals),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "temporal_test_rows": len(temporal_test_df),
        "unseen_test_rows": len(unseen_test),
    }
