"""
ndindex: Automated Spectral Index Discovery

Discovers optimal spectral indices using normalized difference polynomials.
Automatically detects bands, classes, and generates minimal index combinations
for binary classification of multispectral imagery.

Theory:
    Builds on the normalized difference approach (e.g., NDVI = (NIR-Red)/(NIR+Red))
    but systematically explores all band combinations and polynomial terms:
    - Degree 1: ND_ij = (b_i - b_j) / (b_i + b_j + ε)
    - Degree 2: (ND_ij)² and ND_ij × ND_kl

Output:
    - Optimal spectral indices for target accuracy
    - Publication-ready equations with absorbed standardization
    - Performance visualizations

License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
import os
import re
from pathlib import Path

warnings.filterwarnings('ignore')

# Version info
__version__ = "1.0.0"


# ============================================================================
# USER CONFIGURATION
# ============================================================================

def get_user_configuration():
    """
    Interactive configuration from user.
    
    Collects all parameters needed for spectral index discovery:
    - Input CSV files with spectral data
    - Band selection (auto or manual)
    - Analysis parameters (max indices, test split, accuracy threshold)
    
    Returns:
        dict: Configuration dictionary or None if cancelled
    """
    print("=" * 70)
    print("ndindex: SPECTRAL INDEX DISCOVERY")
    print("=" * 70)

    # Get CSV files
    print("\nEnter CSV file paths (one per line, empty line to finish):")
    print("  (Tip: Paste path with or without quotes)")
    csv_files = []
    while True:
        file_path = input(f"  File {len(csv_files) + 1}: ").strip()

        # Remove quotes if present
        file_path = file_path.strip('"').strip("'").strip('r"').strip("r'")

        if not file_path:
            break
        if not os.path.exists(file_path):
            print(f"    ⚠ File not found: {file_path}")
            continue
        csv_files.append(file_path)
        print(f"    ✓ Added: {os.path.basename(file_path)}")

    if not csv_files:
        print("Error: No valid CSV files provided!")
        return None

    print(f"\n✓ Loaded {len(csv_files)} file(s)")

    # Get output directory
    output_dir = input("\nOutput directory (default: current directory): ").strip()
    output_dir = output_dir.strip('"').strip("'")  # Remove quotes
    if not output_dir:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Band selection method
    print("\n" + "=" * 70)
    print("BAND SELECTION")
    print("=" * 70)
    print("Choose how to select spectral bands:")
    print("  1. Auto-detect (finds all columns like b1, b2, b3, ...)")
    print("  2. Manual selection (you specify which columns to use)")

    while True:
        band_method = input("\nSelect method (1 or 2): ").strip()
        if band_method in ['1', '2']:
            break
        print("  ⚠ Please enter 1 or 2")

    band_columns = None
    if band_method == '2':
        print("\nEnter band column names (comma-separated):")
        print("  Example: b1, b2, b3, b4, b5, b6, b7, b8, b9, b10")
        band_input = input("  Bands: ").strip()
        band_columns = [col.strip() for col in band_input.split(',') if col.strip()]

        if not band_columns:
            print("  ⚠ No bands specified, will use auto-detect")
            band_columns = None
        else:
            print(f"  ✓ Will use {len(band_columns)} bands: {', '.join(band_columns)}")

    # Get max indices to search
    while True:
        try:
            max_indices = int(input("\nMaximum number of indices to search (default: 10): ").strip() or "10")
            if max_indices > 0:
                break
            print("  ⚠ Please enter a positive number")
        except ValueError:
            print("  ⚠ Please enter a valid number")

    # Get target column name
    target_col = input("\nTarget column name (default: 'Type'): ").strip()
    if not target_col:
        target_col = "Type"

    # Test split percentage
    while True:
        try:
            test_size = float(input("\nTest set percentage (default: 30): ").strip() or "30")
            test_size = test_size / 100.0
            if 0 < test_size < 1:
                break
            print("  ⚠ Please enter a value between 0 and 100")
        except ValueError:
            print("  ⚠ Please enter a valid number")

    # Accuracy threshold
    while True:
        try:
            acc_threshold = float(input("\nMinimum accuracy threshold % (default: 85): ").strip() or "85")
            acc_threshold = acc_threshold / 100.0
            if 0 < acc_threshold < 1:
                break
            print("  ⚠ Please enter a value between 0 and 100")
        except ValueError:
            print("  ⚠ Please enter a valid number")

    config = {
        'csv_files': csv_files,
        'output_dir': output_dir,
        'band_columns': band_columns,  # None for auto-detect, list for manual
        'max_indices': max_indices,
        'target_col': target_col,
        'test_size': test_size,
        'acc_threshold': acc_threshold
    }

    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    for key, value in config.items():
        if key == 'csv_files':
            print(f"{key}: {len(value)} file(s)")
            for f in value:
                print(f"    - {os.path.basename(f)}")
        elif key == 'band_columns':
            if value is None:
                print(f"{key}: Auto-detect")
            else:
                print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    print("=" * 70)

    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled.")
        return None

    return config


# ============================================================================
# DATA LOADING AND BAND DETECTION
# ============================================================================

def detect_spectral_bands(df):
    """
    Automatically detect spectral band columns.
    
    Looks for columns matching pattern b1, b2, ..., b12, etc.
    Case-insensitive matching.
    
    Args:
        df: pandas DataFrame with spectral data
        
    Returns:
        list: Sorted list of band column names
    """
    band_pattern = re.compile(r'^b\d+$', re.IGNORECASE)
    bands = [col for col in df.columns if band_pattern.match(col)]
    bands = sorted(bands, key=lambda x: int(x[1:]))  # Sort numerically
    return bands


def load_and_merge_data(csv_files, target_col, band_columns=None):
    """
    Load and merge multiple CSV files.
    
    Args:
        csv_files: List of CSV file paths
        target_col: Name of target/class column
        band_columns: Optional list of band columns (None for auto-detect)
        
    Returns:
        tuple: (DataFrame, list of bands, list of classes) or (None, None, None) on error
    """
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    all_dataframes = []
    for i, file_path in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file_path)
            print(f"✓ File {i}: {len(df)} samples, {len(df.columns)} columns")
            all_dataframes.append(df)
        except Exception as e:
            print(f"✗ Error loading file {i}: {e}")
            return None, None, None

    # Merge all dataframes
    df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✓ Total samples: {len(df)}")

    # Detect or use specified bands
    if band_columns is None:
        # Auto-detect
        bands = detect_spectral_bands(df)
        if not bands:
            print("✗ Error: No spectral bands detected (expected columns like b1, b2, b3, ...)")
            return None, None, None
        print(f"✓ Auto-detected {len(bands)} spectral bands: {', '.join(bands)}")
    else:
        # Use manually specified bands
        bands = []
        missing_bands = []
        for band in band_columns:
            if band in df.columns:
                bands.append(band)
            else:
                missing_bands.append(band)

        if missing_bands:
            print(f"⚠ Warning: {len(missing_bands)} bands not found in data: {', '.join(missing_bands)}")

        if not bands:
            print("✗ Error: None of the specified bands were found in the data")
            print(f"  Available columns: {', '.join(df.columns)}")
            return None, None, None

        print(f"✓ Using {len(bands)} specified bands: {', '.join(bands)}")

    # Check target column
    if target_col not in df.columns:
        print(f"✗ Error: Target column '{target_col}' not found in data")
        print(f"  Available columns: {', '.join(df.columns)}")
        return None, None, None

    # Detect classes
    classes = df[target_col].unique()
    if len(classes) != 2:
        print(f"✗ Error: Expected 2 classes, found {len(classes)}: {classes}")
        return None, None, None

    print(f"✓ Detected 2 classes: '{classes[0]}' and '{classes[1]}'")
    print(f"  Class distribution:")
    for cls in classes:
        count = (df[target_col] == cls).sum()
        print(f"    {cls}: {count} samples ({count / len(df) * 100:.1f}%)")

    return df, bands, classes


# ============================================================================
# INDEX GENERATION
# ============================================================================

def create_normalized_difference_indices(df, bands):
    """
    Create all normalized difference polynomial indices.
    
    Generates spectral indices similar to NDVI but explores all possible
    band combinations and polynomial terms up to degree 2:
    - Degree 1: ND_ij = (b_i - b_j) / (b_i + b_j + ε)
    - Degree 2: (ND_ij)² and ND_ij × ND_kl
    
    Args:
        df: DataFrame with spectral band data
        bands: List of band column names
        
    Returns:
        tuple: (DataFrame with indices added, list of index names)
    """
    print("\n" + "=" * 70)
    print("INDEX GENERATION")
    print("=" * 70)

    # Convert bands to numeric
    for band in bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    all_indices = {}
    nd_indices = {}

    # Degree 1: Normalized differences (like NDVI)
    print("Creating degree 1 indices (normalized differences)...")
    for band_i, band_j in combinations(bands, 2):
        nd_name = f"ND_{band_i}_{band_j}"
        # ε = 1e-10 for numerical stability when b_i + b_j ≈ 0
        nd_value = (df[band_i] - df[band_j]) / (df[band_i] + df[band_j] + 1e-10)
        all_indices[nd_name] = nd_value
        nd_indices[nd_name] = nd_value

    print(f"  ✓ Created {len(nd_indices)} ND indices")

    # Degree 2: Squared indices
    print("Creating degree 2 indices (squares)...")
    degree2_squares = {}
    for nd_name, nd_value in nd_indices.items():
        square_name = f"{nd_name}_sq"
        degree2_squares[square_name] = nd_value ** 2

    print(f"  ✓ Created {len(degree2_squares)} squared indices")

    # Degree 2: Product indices
    print("Creating degree 2 indices (products)...")
    degree2_products = {}
    nd_list = list(nd_indices.items())
    for i, (nd_name1, nd_val1) in enumerate(nd_list):
        for nd_name2, nd_val2 in nd_list[i + 1:]:
            product_name = f"{nd_name1}_{nd_name2}_prod"
            degree2_products[product_name] = nd_val1 * nd_val2

    print(f"  ✓ Created {len(degree2_products)} product indices")

    # Combine all indices
    all_indices = {**all_indices, **degree2_squares, **degree2_products}
    df = pd.concat([df, pd.DataFrame(all_indices)], axis=1)

    total_indices = len(all_indices)
    print(f"\n✓ TOTAL INDEX SPACE: {total_indices} indices")
    print(f"    Degree 1 (ND): {len(nd_indices)}")
    print(f"    Degree 2 (squares): {len(degree2_squares)}")
    print(f"    Degree 2 (products): {len(degree2_products)}")

    return df, list(all_indices.keys())


# ============================================================================
# INDEX SELECTION AND EVALUATION
# ============================================================================

def run_index_selection(X_train, X_test, y_train, y_test, index_range, all_indices):
    """
    Run comprehensive index selection analysis.
    
    Tests two feature selection methods and picks the best:
    - RFE (Recursive Feature Elimination): Wrapper method using SVM coefficients
    - SelectKBest: Filter method using ANOVA F-statistic
    
    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test labels
        index_range: List of index counts to test (e.g., [1, 2, 3, ...])
        all_indices: List of all index names
        
    Returns:
        list: Results for each index count
    """
    print("\n" + "=" * 70)
    print(f"INDEX SELECTION: Testing 1 to {max(index_range)} indices")
    print("=" * 70)

    all_results = []

    for k in index_range:
        # Method 1: RFE (Recursive Feature Elimination)
        svm_rfe = SVC(kernel='linear', C=1.0, random_state=42)
        rfe = RFE(estimator=svm_rfe, n_features_to_select=k, step=1)
        rfe.fit(X_train, y_train)

        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        svm_rfe.fit(X_train_rfe, y_train)
        rfe_test_acc = accuracy_score(y_test, svm_rfe.predict(X_test_rfe))
        rfe_train_acc = accuracy_score(y_train, svm_rfe.predict(X_train_rfe))
        rfe_indices = np.where(rfe.support_)[0]

        # Method 2: SelectKBest (ANOVA F-statistic)
        selector = SelectKBest(f_classif, k=k)
        X_train_kb = selector.fit_transform(X_train, y_train)
        X_test_kb = selector.transform(X_test)

        svm_kb = SVC(kernel='linear', C=1.0, random_state=42)
        svm_kb.fit(X_train_kb, y_train)
        kb_test_acc = accuracy_score(y_test, svm_kb.predict(X_test_kb))
        kb_train_acc = accuracy_score(y_train, svm_kb.predict(X_train_kb))
        kb_indices = selector.get_support(indices=True)

        # Pick better method based on test accuracy
        if rfe_test_acc >= kb_test_acc:
            best_method = 'RFE'
            test_acc = rfe_test_acc
            train_acc = rfe_train_acc
            selected_indices = rfe_indices
            best_svm = svm_rfe
            X_train_selected = X_train_rfe
            X_test_selected = X_test_rfe
        else:
            best_method = 'SelectKBest'
            test_acc = kb_test_acc
            train_acc = kb_train_acc
            selected_indices = kb_indices
            best_svm = svm_kb
            X_train_selected = X_train_kb
            X_test_selected = X_test_kb

        all_results.append({
            'n_indices': k,
            'method': best_method,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc,
            'indices': selected_indices,
            'svm': best_svm,
            'X_train': X_train_selected,
            'X_test': X_test_selected
        })

        status = "✓" if test_acc >= 0.85 else "✗"
        print(
            f"{k:2d} indices ({best_method:11s}): Test: {test_acc:.4f} | Train: {train_acc:.4f} | Gap: {train_acc - test_acc:.4f} {status}")

    return all_results


# ============================================================================
# RESULTS ANALYSIS AND EXPORT
# ============================================================================

def analyze_and_export_results(all_results, all_indices, index_means, index_stds,
                               config, classes, bands):
    """
    Analyze results and generate reports.
    
    Identifies key models:
    - Minimum indices achieving target accuracy
    - Sweet spot (diminishing returns)
    - Best overall accuracy
    
    Exports equations and visualizations.
    """
    results_df = pd.DataFrame(all_results)
    output_dir = config['output_dir']

    # Find key models
    above_threshold = results_df[results_df['test_acc'] >= config['acc_threshold']]

    print("\n" + "=" * 70)
    print("INDEX DISCOVERY RESULTS")
    print("=" * 70)

    # Minimum indices above threshold
    min_indices = None
    if len(above_threshold) > 0:
        min_indices = above_threshold['n_indices'].min()
        min_result = above_threshold[above_threshold['n_indices'] == min_indices].iloc[0]
        print(f"\n⭐ MINIMUM INDICES (≥{config['acc_threshold'] * 100:.0f}% accuracy): {int(min_indices)}")
        print(f"   Test Accuracy: {min_result['test_acc']:.4f} ({min_result['test_acc'] * 100:.2f}%)")
        print(f"   Train-Test Gap: {min_result['gap']:.4f}")
        
        # Show the selected indices
        print(f"   Selected indices:")
        for idx in min_result['indices']:
            print(f"      - {all_indices[idx]}")

    # Sweet spot (diminishing returns)
    sweet_spot = None
    for i in range(1, len(results_df)):
        prev_acc = results_df.loc[i - 1, 'test_acc']
        curr_acc = results_df.loc[i, 'test_acc']
        improvement = curr_acc - prev_acc

        if sweet_spot is None and improvement < 0.005 and curr_acc >= 0.95:
            sweet_spot = results_df.loc[i - 1, 'n_indices']

    if sweet_spot is not None:
        sweet_result = results_df[results_df['n_indices'] == sweet_spot].iloc[0]
        print(f"\n⭐ SWEET SPOT (diminishing returns): {int(sweet_spot)} indices")
        print(f"   Test Accuracy: {sweet_result['test_acc']:.4f} ({sweet_result['test_acc'] * 100:.2f}%)")

    # Best accuracy
    best_idx = results_df['test_acc'].idxmax()
    best_result = results_df.loc[best_idx]
    print(f"\n⭐ BEST ACCURACY: {int(best_result['n_indices'])} indices")
    print(f"   Test Accuracy: {best_result['test_acc']:.4f} ({best_result['test_acc'] * 100:.2f}%)")

    # Export equations
    export_index_equations(all_results, all_indices, index_means, index_stds,
                           output_dir, classes, bands, config)

    # Generate plots
    generate_plots(results_df, min_indices,
                   sweet_spot, output_dir, config)

    return results_df


def export_index_equations(all_results, all_indices, index_means, index_stds,
                           output_dir, classes, bands, config):
    """
    Export spectral index equations.
    
    Generates publication-ready equations with standardization absorbed
    into coefficients, allowing direct application to raw band values.
    """
    equations_file = os.path.join(output_dir, "spectral_indices.txt")

    with open(equations_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SPECTRAL INDEX EQUATIONS\n")
        f.write("Generated by ndindex\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nClasses: '{classes[0]}' vs '{classes[1]}'\n")
        f.write(f"Decision Rule: f(x) > 0 => {classes[0]} | f(x) <= 0 => {classes[1]}\n")
        f.write(f"\nSpectral bands used: {', '.join(bands)}\n")
        f.write(f"Total index space: {len(all_indices)}\n")
        f.write("\nIndex formulation (degree ≤ 2 polynomials):\n")
        f.write("  - Degree 1: ND_ij = (b_i - b_j) / (b_i + b_j + ε), ε = 10^-10\n")
        f.write("  - Degree 2: (ND_ij)² and ND_ij × ND_kl\n")
        f.write("\nNote: Standardization absorbed into coefficients.\n")
        f.write("Apply directly to raw band values.\n\n")

        for result in all_results:
            k = result['n_indices']
            svm = result['svm']
            indices_idx = result['indices']
            index_names = [all_indices[i] for i in indices_idx]

            coefs_std = svm.coef_[0]
            intercept_std = svm.intercept_[0]
            test_acc = result['test_acc']

            # Absorb standardization into coefficients
            means = index_means[indices_idx]
            stds = index_stds[indices_idx]
            new_coefs = coefs_std / stds
            new_intercept = intercept_std - np.sum(coefs_std * means / stds)

            f.write(f"\n{'=' * 70}\n")
            f.write(f"{k}-INDEX MODEL (Test Accuracy: {test_acc:.4f})\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"f(x) = {new_intercept:.6f}\n")

            for name, coef in zip(index_names, new_coefs):
                f.write(f"       {coef:+.6f} * {name}\n")

    print(f"\n✓ Index equations saved: {equations_file}")


def generate_plots(results_df, min_indices, sweet_spot, output_dir, config):
    """
    Generate visualization plots for index selection analysis.
    
    Creates 4-panel figure:
    - Test accuracy vs number of indices
    - Overfitting gap analysis
    - Train vs test accuracy comparison
    - Marginal improvement per index
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Spectral Index Discovery Results', fontsize=16, fontweight='bold', y=1.02)

    # Plot 1: Test accuracy
    ax1 = axes[0, 0]
    ax1.plot(results_df['n_indices'], results_df['test_acc'], 'o-', linewidth=2, markersize=6)
    ax1.axhline(y=config['acc_threshold'], color='red', linestyle='--',
                label=f"{config['acc_threshold'] * 100:.0f}% threshold")
    if sweet_spot:
        ax1.axvline(x=sweet_spot, color='green', linestyle='--', label=f'Sweet spot ({int(sweet_spot)})')
    if min_indices:
        ax1.axvline(x=min_indices, color='orange', linestyle='--', label=f'Minimum ({int(min_indices)})')
    ax1.set_xlabel('Number of Indices', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Number of Indices', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Overfitting gap
    ax2 = axes[0, 1]
    colors = ['green' if gap < 0.02 else 'orange' if gap < 0.05 else 'red'
              for gap in results_df['gap']]
    ax2.bar(results_df['n_indices'], results_df['gap'], color=colors, alpha=0.7)
    ax2.axhline(y=0.02, color='orange', linestyle='--', label='2% threshold')
    ax2.set_xlabel('Number of Indices', fontsize=12)
    ax2.set_ylabel('Train-Test Gap', fontsize=12)
    ax2.set_title('Overfitting Gap', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Train vs Test
    ax3 = axes[1, 0]
    ax3.plot(results_df['n_indices'], results_df['train_acc'], 'o-', label='Train', linewidth=2)
    ax3.plot(results_df['n_indices'], results_df['test_acc'], 's-', label='Test', linewidth=2)
    ax3.set_xlabel('Number of Indices', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Marginal improvements
    ax4 = axes[1, 1]
    improvements = [0] + [results_df.loc[i, 'test_acc'] - results_df.loc[i - 1, 'test_acc']
                         for i in range(1, len(results_df))]
    ax4.bar(results_df['n_indices'], improvements, alpha=0.7, color='steelblue')
    ax4.axhline(y=0.005, color='red', linestyle='--', label='0.5% threshold')
    ax4.set_xlabel('Number of Indices', fontsize=12)
    ax4.set_ylabel('Accuracy Improvement', fontsize=12)
    ax4.set_title('Marginal Improvement per Index', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "index_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved: {plot_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Get user configuration
    2. Load and validate data
    3. Generate normalized difference polynomial indices
    4. Run index selection with multiple methods
    5. Analyze and export results
    """
    print("\n" + "=" * 70)
    print("ndindex v" + __version__)
    print("Automated Spectral Index Discovery")
    print("=" * 70)
    
    # Get configuration from user
    config = get_user_configuration()
    if config is None:
        return

    # Load and prepare data
    df, bands, classes = load_and_merge_data(
        config['csv_files'],
        config['target_col'],
        config['band_columns']
    )
    if df is None:
        return

    # Create indices
    df, all_indices = create_normalized_difference_indices(df, bands)

    # Prepare training data
    X, y = df[all_indices].values, df[config['target_col']].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X_clean, y_clean = X[mask], y[mask]

    print(f"\n✓ Valid samples after cleaning: {len(X_clean)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=config['test_size'], random_state=42, stratify=y_clean
    )

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Standardize (for SVM training)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    index_means = scaler.mean_
    index_stds = scaler.scale_

    # Run index selection
    index_range = list(range(1, min(config['max_indices'] + 1, len(all_indices) + 1)))
    all_results = run_index_selection(X_train_scaled, X_test_scaled, y_train, y_test,
                                      index_range, all_indices)

    # Analyze and export results
    analyze_and_export_results(all_results, all_indices, index_means, index_stds,
                               config, classes, bands)

    print("\n" + "=" * 70)
    print("INDEX DISCOVERY COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {os.path.join(config['output_dir'], 'spectral_indices.txt')}")
    print(f"  - {os.path.join(config['output_dir'], 'index_analysis.png')}")
    print("\nThank you for using ndindex!")


if __name__ == "__main__":
    main()
