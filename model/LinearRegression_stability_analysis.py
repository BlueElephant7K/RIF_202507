#!/usr/bin/env python3
"""
Linear Regression Stability Analysis
Runs comprehensive stability analysis for both Y1 and Y2 outcomes using Linear Regression variants
"""
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
sys.path.append(root_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, BayesianRidge
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, classification_report
)
from scipy.stats import pearsonr
from collections import Counter, defaultdict
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

def run_stability_analysis(selected_outcome='Y1'):
    """Run stability analysis for specified outcome using Linear Regression"""
    print(f"\n{'='*60}")
    print(f"STARTING LINEAR REGRESSION STABILITY ANALYSIS FOR {selected_outcome}")
    print(f"{'='*60}")
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('raw_data/filtered_已经log2.csv')
    
    # Rename clinical columns to English
    column_mapping = {
        '临床妊娠结局': 'Y1',  # Clinical pregnancy outcome
        '活产结局': 'Y2',      # Live birth outcome  
        '体重指数': 'BMI',      # Body mass index
        '基础内分泌FSH': 'FSH', # Basal FSH
        '基础内分泌AMH': 'AMH', # Basal AMH
        '移植胚胎数': 'Embryo_Count'  # Number of transferred embryos
    }
    
    df = df.rename(columns=column_mapping)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Y1 distribution: {df['Y1'].value_counts().to_dict()}")
    print(f"Y2 distribution: {df['Y2'].value_counts().to_dict()}")
    
    # Configuration
    SELECTED_OUTCOME = selected_outcome
    N_FEATURES = 1000
    N_ITERATIONS = 20  # Reduced for faster execution
    TEST_SIZE = 0.2
    
    print(f"\nConfiguration:")
    print(f"  Selected outcome: {SELECTED_OUTCOME}")
    print(f"  Features to select: {N_FEATURES}")
    print(f"  Stability iterations: {N_ITERATIONS}")
    print(f"  Test set size: {TEST_SIZE}")
    
    # Prepare data
    y = df[SELECTED_OUTCOME].copy().astype(float)  # Convert to float for regression
    clinical_cols = ['Y1', 'Y2', 'BMI', 'FSH', 'AMH', 'Embryo_Count']
    gene_cols = [col for col in df.columns if col not in clinical_cols]
    X = df[gene_cols].select_dtypes(include=[np.number]).copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.mode()[0])
    
    print(f"\nData prepared:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(y)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    print(f"  Target as regression: min={y.min()}, max={y.max()}, mean={y.mean():.3f}")
    
    # Initialize storage for stability results
    stability_results = {
        'iteration': [],
        'model_name': [],
        # Regression metrics
        'mse': [],
        'mae': [],
        'r2_score': [],
        'rmse': [],
        # Classification metrics (convert predictions back to binary)
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc_roc': [],
        'selected_features': [],
        'feature_scores': [],
        'predictions': [],
        'predictions_binary': [],
        'test_indices': [],
        'feature_coefficients': []
    }
    
    # Linear Regression models to test
    models = {
        'Linear_Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
        'Huber': HuberRegressor(epsilon=1.35, max_iter=1000),
        'Bayesian_Ridge': BayesianRidge(compute_score=True)
    }
    
    print(f"\nStarting stability testing with {N_ITERATIONS} iterations...")
    print("Testing 6 Linear Regression variants: Linear, Ridge, Lasso, ElasticNet, Huber, Bayesian Ridge")
    print("This may take a few minutes...")
    
    # Main stability testing loop
    for iteration in range(N_ITERATIONS):
        if (iteration + 1) % 5 == 0:
            print(f"  Completed {iteration + 1}/{N_ITERATIONS} iterations")
        
        # CORRECT APPROACH: Split data first, then feature selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=iteration, stratify=y.astype(int)
        )
        
        # Feature selection ONLY on training data
        selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature information
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]
        
        # Scale features (important for regularized regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Test each model
        for model_name, model in models.items():
            # Clone the model for this iteration
            model_copy = type(model)(**model.get_params())
            
            try:
                # Train model
                model_copy.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model_copy.predict(X_test_scaled)
                
                # Convert predictions to binary for classification metrics
                y_pred_binary = (y_pred >= 0.5).astype(int)
                y_test_binary = y_test.astype(int)
                
                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Calculate classification metrics
                accuracy = accuracy_score(y_test_binary, y_pred_binary)
                precision = precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
                recall = recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
                f1 = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
                
                # AUC-ROC using continuous predictions
                if len(np.unique(y_test_binary)) == 2:
                    # Clip predictions to valid probability range
                    y_pred_clipped = np.clip(y_pred, 0, 1)
                    auc = roc_auc_score(y_test_binary, y_pred_clipped)
                else:
                    auc = None
                
                # Get feature coefficients for interpretability
                if hasattr(model_copy, 'coef_'):
                    feature_coefficients = model_copy.coef_
                elif hasattr(model_copy, 'coef_'):  # Bayesian Ridge
                    feature_coefficients = model_copy.coef_
                else:
                    feature_coefficients = None
                
                # Store results
                stability_results['iteration'].append(iteration)
                stability_results['model_name'].append(model_name)
                stability_results['mse'].append(mse)
                stability_results['mae'].append(mae)
                stability_results['r2_score'].append(r2)
                stability_results['rmse'].append(rmse)
                stability_results['accuracy'].append(accuracy)
                stability_results['precision'].append(precision)
                stability_results['recall'].append(recall)
                stability_results['f1_score'].append(f1)
                stability_results['auc_roc'].append(auc)
                stability_results['selected_features'].append(list(selected_features))
                stability_results['feature_scores'].append(list(feature_scores))
                stability_results['predictions'].append(list(y_pred))
                stability_results['predictions_binary'].append(list(y_pred_binary))
                stability_results['test_indices'].append(list(X_test.index))
                stability_results['feature_coefficients'].append(feature_coefficients.tolist() if feature_coefficients is not None else None)
                
            except Exception as e:
                print(f"  Warning: {model_name} failed in iteration {iteration}: {e}")
                # Store NaN values for failed runs
                stability_results['iteration'].append(iteration)
                stability_results['model_name'].append(model_name)
                for key in ['mse', 'mae', 'r2_score', 'rmse', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                    stability_results[key].append(np.nan)
                stability_results['selected_features'].append(list(selected_features))
                stability_results['feature_scores'].append(list(feature_scores))
                stability_results['predictions'].append([np.nan] * len(y_test))
                stability_results['predictions_binary'].append([np.nan] * len(y_test))
                stability_results['test_indices'].append(list(X_test.index))
                stability_results['feature_coefficients'].append(None)
    
    print(f"\nStability testing completed!")
    print(f"Total experiments: {len(stability_results['iteration'])}")
    
    # Process results
    results_df = pd.DataFrame({
        'iteration': stability_results['iteration'],
        'model_name': stability_results['model_name'],
        'mse': stability_results['mse'],
        'mae': stability_results['mae'],
        'r2_score': stability_results['r2_score'],
        'rmse': stability_results['rmse'],
        'accuracy': stability_results['accuracy'],
        'precision': stability_results['precision'],
        'recall': stability_results['recall'],
        'f1_score': stability_results['f1_score'],
        'auc_roc': stability_results['auc_roc']
    })
    
    print("\nPerformance Summary by Model:")
    print("=" * 60)
    
    # Calculate summary statistics for each model
    summary_stats = results_df.groupby('model_name').agg({
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2_score': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'auc_roc': ['mean', 'std']
    }).round(4)
    
    print(summary_stats)
    
    # Best performing model (by R²)
    best_model_r2 = results_df.groupby('model_name')['r2_score'].mean().idxmax()
    best_r2 = results_df.groupby('model_name')['r2_score'].mean().max()
    
    # Best performing model (by accuracy)
    best_model_acc = results_df.groupby('model_name')['accuracy'].mean().idxmax()
    best_acc = results_df.groupby('model_name')['accuracy'].mean().max()
    
    print(f"\nBest performing model (R²): {best_model_r2} (R² = {best_r2:.4f})")
    print(f"Best performing model (Accuracy): {best_model_acc} (Acc = {best_acc:.4f})")
    
    # Feature stability analysis
    all_features = set()
    feature_frequency = Counter()
    
    total_possible_selections = len(models) * N_ITERATIONS
    
    for features in stability_results['selected_features']:
        all_features.update(features)
        feature_frequency.update(features)
    
    print(f"\nFeature Selection Stability Analysis:")
    print("=" * 50)
    print(f"Total unique features ever selected: {len(all_features)}")
    print(f"Total possible selections per feature: {total_possible_selections}")
    print(f"Features selected in all model iterations: {sum(1 for count in feature_frequency.values() if count == total_possible_selections)}")
    print(f"Features selected in ≥80% of model iterations: {sum(1 for count in feature_frequency.values() if count >= 0.8 * total_possible_selections)}")
    print(f"Features selected in ≥50% of model iterations: {sum(1 for count in feature_frequency.values() if count >= 0.5 * total_possible_selections)}")
    
    # Create feature stability dataframe
    feature_stability_df = pd.DataFrame([
        {
            'feature': feature, 
            'frequency': count, 
            'percentage': (count / total_possible_selections) * 100
        }
        for feature, count in feature_frequency.items()
    ]).sort_values('frequency', ascending=False)
    
    # Top stable features
    print(f"\nTop 20 Most Stable Features:")
    print("=" * 30)
    top_20_stable = feature_stability_df.head(20)
    for _, row in top_20_stable.iterrows():
        print(f"{row['feature']:15} | {row['frequency']:3}/{total_possible_selections} ({row['percentage']:5.1f}%)")
    
    # Core biomarkers
    core_biomarkers = feature_stability_df[feature_stability_df['percentage'] >= 80]
    print(f"\nCore Biomarkers (≥80% selection rate): {len(core_biomarkers)} genes")
    if len(core_biomarkers) > 0:
        print("Core genes:", list(core_biomarkers['feature'].head(10)))
    
    # Feature coefficient analysis
    print(f"\nFeature Coefficient Analysis:")
    print("=" * 50)
    
    # Analyze coefficients for regularized models
    coef_models = ['Ridge', 'Lasso', 'ElasticNet', 'Bayesian_Ridge']
    coefficient_analysis = {}
    
    for model_name in coef_models:
        model_results = [(i, coefs, features) for i, (model, coefs, features) in 
                        enumerate(zip(stability_results['model_name'], 
                                     stability_results['feature_coefficients'],
                                     stability_results['selected_features'])) 
                        if model == model_name and coefs is not None]
        
        if model_results:
            # Calculate average coefficient magnitudes
            coef_analysis = defaultdict(list)
            for iteration, coefficients, features in model_results:
                for feature, coef in zip(features, coefficients):
                    coef_analysis[feature].append(abs(coef))
            
            # Get top features by coefficient magnitude
            stable_features_with_coefs = []
            for _, row in feature_stability_df.head(20).iterrows():
                feature = row['feature']
                if feature in coef_analysis and len(coef_analysis[feature]) > 0:
                    avg_coef_magnitude = np.mean(coef_analysis[feature])
                    stable_features_with_coefs.append((feature, row['percentage'], avg_coef_magnitude))
            
            stable_features_with_coefs.sort(key=lambda x: x[2], reverse=True)
            coefficient_analysis[model_name] = stable_features_with_coefs
            
            print(f"\nTop 10 features by coefficient magnitude ({model_name}):")
            for i, (feature, stability, coef_mag) in enumerate(stable_features_with_coefs[:10], 1):
                print(f"  {i:2d}. {feature:15} | Stability: {stability:5.1f}% | Avg |coef|: {coef_mag:.4f}")
    
    # Model ranking analysis
    models_list = results_df['model_name'].unique()
    
    # Combined score (R² - instability penalty)
    combined_scores = []
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        mean_r2 = model_data['r2_score'].mean()
        cv_r2 = model_data['r2_score'].std() / abs(mean_r2) if mean_r2 != 0 else float('inf')
        combined_score = mean_r2 - cv_r2  # Penalize instability
        combined_scores.append((model, combined_score))
    
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nModel Ranking by Combined Score (R² - Stability Penalty):")
    for i, (model, score) in enumerate(combined_scores, 1):
        print(f"  {i}. {model}: {score:.4f}")
    
    # Generate comprehensive summary
    best_overall = combined_scores[0][0]
    best_model_data = results_df[results_df['model_name'] == best_overall]
    overall_mean_r2 = results_df['r2_score'].mean()
    overall_mean_acc = results_df['accuracy'].mean()
    
    if overall_mean_r2 > 0.5:
        regression_quality = "Good"
    elif overall_mean_r2 > 0.3:
        regression_quality = "Moderate"
    else:
        regression_quality = "Poor"
    
    if overall_mean_acc > 0.8:
        classification_quality = "Excellent"
    elif overall_mean_acc > 0.7:
        classification_quality = "Good"
    elif overall_mean_acc > 0.6:
        classification_quality = "Moderate"
    else:
        classification_quality = "Poor"
    
    high_stable = len(feature_stability_df[feature_stability_df['percentage'] >= 80])
    
    print(f"\nCOMPREHENSIVE SUMMARY FOR {SELECTED_OUTCOME}:")
    print("=" * 50)
    print(f"Best Overall Model: {best_overall}")
    print(f"Regression Performance (R²): {best_model_data['r2_score'].mean():.4f} ± {best_model_data['r2_score'].std():.4f}")
    print(f"Classification Performance (Acc): {best_model_data['accuracy'].mean():.4f} ± {best_model_data['accuracy'].std():.4f}")
    print(f"Regression Quality: {regression_quality}")
    print(f"Classification Quality: {classification_quality}")
    print(f"Highly Stable Genes (≥80%): {high_stable}")
    print(f"Top Biomarker: {feature_stability_df.iloc[0]['feature']} ({feature_stability_df.iloc[0]['percentage']:.1f}%)")
    
    # Linear Regression specific insights
    print(f"\nLinear Regression Insights:")
    print(f"  Ridge: L2 regularization, handles multicollinearity")
    print(f"  Lasso: L1 regularization, automatic feature selection")
    print(f"  ElasticNet: Combines L1 and L2 regularization")
    print(f"  Huber: Robust to outliers")
    print(f"  Bayesian Ridge: Provides uncertainty estimates")
    print(f"  Coefficients provide direct feature impact interpretation")
    
    # Generate and save visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories if they don't exist
    result_num_dir = f'result_num/linear_regression'
    result_pic_dir = f'result_pic/linear_regression'
    os.makedirs(result_num_dir, exist_ok=True)
    os.makedirs(result_pic_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # 1. Dual metric performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² boxplot
    r2_data = [results_df[results_df['model_name'] == model]['r2_score'].values for model in models_list]
    axes[0, 0].boxplot(r2_data, labels=[m.replace('_', ' ') for m in models_list])
    axes[0, 0].set_title(f'R² Score Distribution - {SELECTED_OUTCOME}')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Accuracy boxplot
    acc_data = [results_df[results_df['model_name'] == model]['accuracy'].values for model in models_list]
    axes[0, 1].boxplot(acc_data, labels=[m.replace('_', ' ') for m in models_list])
    axes[0, 1].set_title(f'Classification Accuracy - {SELECTED_OUTCOME}')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Regression vs Classification performance scatter
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        axes[1, 0].scatter(model_data['r2_score'], model_data['accuracy'], 
                          alpha=0.7, label=model.replace('_', ' '), s=30)
    
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_ylabel('Classification Accuracy')
    axes[1, 0].set_title(f'Regression vs Classification Performance - {SELECTED_OUTCOME}')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature stability histogram
    axes[1, 1].hist(feature_stability_df['percentage'], bins=15, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(80, color='red', linestyle='--', label='80% threshold')
    axes[1, 1].axvline(50, color='orange', linestyle='--', label='50% threshold')
    axes[1, 1].set_xlabel('Selection Percentage')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title(f'Feature Selection Stability - {SELECTED_OUTCOME}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/performance_analysis_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model comparison metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MSE comparison
    mse_means = [results_df[results_df['model_name'] == model]['mse'].mean() for model in models_list]
    mse_stds = [results_df[results_df['model_name'] == model]['mse'].std() for model in models_list]
    bars1 = axes[0, 0].bar([m.replace('_', ' ') for m in models_list], mse_means, 
                          yerr=mse_stds, alpha=0.8, capsize=5)
    axes[0, 0].set_title(f'Mean Squared Error - {SELECTED_OUTCOME}')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² comparison
    r2_means = [results_df[results_df['model_name'] == model]['r2_score'].mean() for model in models_list]
    r2_stds = [results_df[results_df['model_name'] == model]['r2_score'].std() for model in models_list]
    bars2 = axes[0, 1].bar([m.replace('_', ' ') for m in models_list], r2_means, 
                          yerr=r2_stds, alpha=0.8, capsize=5)
    axes[0, 1].set_title(f'R² Score - {SELECTED_OUTCOME}')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC comparison
    auc_means = [results_df[results_df['model_name'] == model]['auc_roc'].mean() for model in models_list]
    auc_stds = [results_df[results_df['model_name'] == model]['auc_roc'].std() for model in models_list]
    bars3 = axes[0, 2].bar([m.replace('_', ' ') for m in models_list], auc_means, 
                          yerr=auc_stds, alpha=0.8, capsize=5)
    axes[0, 2].set_title(f'AUC-ROC - {SELECTED_OUTCOME}')
    axes[0, 2].set_ylabel('AUC-ROC')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Model stability (CV of R²)
    cv_r2_data = []
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        cv = model_data['r2_score'].std() / abs(model_data['r2_score'].mean()) if model_data['r2_score'].mean() != 0 else 0
        cv_r2_data.append(cv)
    
    bars4 = axes[1, 0].bar([m.replace('_', ' ') for m in models_list], cv_r2_data, alpha=0.8)
    axes[1, 0].set_title(f'R² Stability - {SELECTED_OUTCOME}\\n(Lower = More Stable)')
    axes[1, 0].set_ylabel('Coefficient of Variation')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top 15 stable features
    top_15 = feature_stability_df.head(15)
    axes[1, 1].barh(range(len(top_15)), top_15['percentage'])
    axes[1, 1].set_yticks(range(len(top_15)))
    axes[1, 1].set_yticklabels(top_15['feature'], fontsize=8)
    axes[1, 1].set_xlabel('Selection Percentage')
    axes[1, 1].set_title(f'Top 15 Stable Features - {SELECTED_OUTCOME}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Residual analysis for best model
    best_model_results = results_df[results_df['model_name'] == best_overall]
    # Get predictions from last iteration for residual plot
    if len(stability_results['predictions']) > 0:
        last_predictions = stability_results['predictions'][-1]  # Last iteration
        last_actual = [y.iloc[idx] for idx in stability_results['test_indices'][-1]]
        residuals = np.array(last_actual) - np.array(last_predictions)
        
        axes[1, 2].scatter(last_predictions, residuals, alpha=0.6)
        axes[1, 2].axhline(y=0, color='red', linestyle='--')
        axes[1, 2].set_xlabel('Predicted Values')
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title(f'Residual Plot - {best_overall}')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No residual data\\navailable', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Residual Analysis')
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/model_comparison_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Coefficient analysis for regularized models
    if coefficient_analysis:
        n_models = len(coefficient_analysis)
        fig, axes = plt.subplots(1, min(n_models, 3), figsize=(6*min(n_models, 3), 8))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, coef_data) in enumerate(list(coefficient_analysis.items())[:3]):
            if coef_data:
                top_coef_features = coef_data[:15]  # Top 15 by coefficient magnitude
                features = [f[0] for f in top_coef_features]
                coef_mags = [f[2] for f in top_coef_features]
                
                bars = axes[i].barh(range(len(features)), coef_mags, alpha=0.8)
                axes[i].set_yticks(range(len(features)))
                axes[i].set_yticklabels(features, fontsize=8)
                axes[i].set_xlabel('Average |Coefficient| Magnitude')
                axes[i].set_title(f'{model_name} - Top Features\\n{SELECTED_OUTCOME}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide extra subplots if any
        for j in range(len(coefficient_analysis), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{result_pic_dir}/coefficient_analysis_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {result_pic_dir}/")
    
    # Save results
    
    # Save detailed results
    results_filename = f'{result_num_dir}/stability_analysis_{SELECTED_OUTCOME}_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"\nDetailed results saved to: {results_filename}")
    
    # Save feature stability analysis
    features_filename = f'{result_num_dir}/feature_stability_{SELECTED_OUTCOME}_{timestamp}.csv'
    feature_stability_df.to_csv(features_filename, index=False)
    print(f"Feature stability saved to: {features_filename}")
    
    # Save core biomarkers
    biomarkers_filename = f'{result_num_dir}/core_biomarkers_{SELECTED_OUTCOME}_{timestamp}.csv'
    core_biomarkers.to_csv(biomarkers_filename, index=False)
    print(f"Core biomarkers saved to: {biomarkers_filename}")
    
    # Save coefficient analysis
    if coefficient_analysis:
        for model_name, coef_data in coefficient_analysis.items():
            if coef_data:
                coef_df = pd.DataFrame(coef_data, columns=['feature', 'stability_percentage', 'avg_coefficient_magnitude'])
                coef_filename = f'{result_num_dir}/coefficient_analysis_{model_name}_{SELECTED_OUTCOME}_{timestamp}.csv'
                coef_df.to_csv(coef_filename, index=False)
                print(f"Coefficient analysis ({model_name}) saved to: {coef_filename}")
    
    # Save complete stability data
    stability_data_filename = f'{result_num_dir}/complete_stability_data_{SELECTED_OUTCOME}_{timestamp}.pkl'
    with open(stability_data_filename, 'wb') as f:
        pickle.dump({
            'config': {
                'outcome': SELECTED_OUTCOME,
                'n_features': N_FEATURES,
                'n_iterations': N_ITERATIONS,
                'test_size': TEST_SIZE,
                'models': list(models.keys())
            },
            'results': stability_results,
            'summary': {
                'best_model': best_overall,
                'model_rankings': combined_scores,
                'feature_stability': feature_stability_df,
                'performance_summary': summary_stats,
                'coefficient_analysis': coefficient_analysis
            }
        }, f)
    
    print(f"Complete stability data saved to: {stability_data_filename}")
    
    # Save summary report
    report_filename = f'{result_num_dir}/stability_report_{SELECTED_OUTCOME}_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(f"LINEAR REGRESSION STABILITY ANALYSIS REPORT\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Outcome: {SELECTED_OUTCOME}\\n")
        f.write(f"Iterations: {N_ITERATIONS}\\n")
        f.write(f"Models: Linear, Ridge, Lasso, ElasticNet, Huber, Bayesian Ridge\\n\\n")
        
        f.write(f"BEST MODEL: {best_overall}\\n")
        f.write(f"R² Performance: {best_model_data['r2_score'].mean():.4f} ± {best_model_data['r2_score'].std():.4f}\\n")
        f.write(f"Classification Performance: {best_model_data['accuracy'].mean():.4f} ± {best_model_data['accuracy'].std():.4f}\\n\\n")
        
        f.write(f"TOP 10 STABLE BIOMARKERS:\\n")
        for i, (_, row) in enumerate(feature_stability_df.head(10).iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:15} ({row['percentage']:5.1f}%)\\n")
        
        f.write(f"\\nPERFORMANCE SUMMARY:\\n")
        f.write(f"Regression Quality: {regression_quality}\\n")
        f.write(f"Classification Quality: {classification_quality}\\n")
        f.write(f"High stability genes (≥80%): {high_stable}\\n")
        
        if coefficient_analysis:
            f.write(f"\\nTOP COEFFICIENT FEATURES:\\n")
            for model_name, coef_data in coefficient_analysis.items():
                if coef_data:
                    f.write(f"\\n{model_name}:\\n")
                    for i, (feature, stability, coef_mag) in enumerate(coef_data[:5], 1):
                        f.write(f"  {i}. {feature:15} | Stability: {stability:5.1f}% | |coef|: {coef_mag:.4f}\\n")
    
    print(f"Summary report saved to: {report_filename}")
    
    print(f"\n{'='*50}")
    print(f"LINEAR REGRESSION ANALYSIS FOR {SELECTED_OUTCOME} COMPLETED SUCCESSFULLY!")
    print(f"Results directory: {result_num_dir}")
    print(f"Figures directory: {result_pic_dir}")
    print(f"Analysis timestamp: {timestamp}")
    print(f"{'='*50}")
    
    return {
        'outcome': SELECTED_OUTCOME,
        'best_model': best_overall,
        'r2_performance': best_model_data['r2_score'].mean(),
        'classification_performance': best_model_data['accuracy'].mean(),
        'r2_stability': best_model_data['r2_score'].std(),
        'regression_quality': regression_quality,
        'classification_quality': classification_quality,
        'high_stable_genes': high_stable,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    print("LINEAR REGRESSION STABILITY ANALYSIS")
    print("Running analysis for both Y1 and Y2 outcomes...")
    
    # Run for Y1 (Clinical pregnancy outcome)
    results_y1 = run_stability_analysis('Y1')
    
    # Run for Y2 (Live birth outcome)  
    results_y2 = run_stability_analysis('Y2')
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON SUMMARY - LINEAR REGRESSION")
    print(f"{'='*70}")
    print(f"Y1 (Clinical Pregnancy):")
    print(f"  Best Model: {results_y1['best_model']}")
    print(f"  R² Performance: {results_y1['r2_performance']:.4f} ± {results_y1['r2_stability']:.4f}")
    print(f"  Classification Acc: {results_y1['classification_performance']:.4f}")
    print(f"  Regression Quality: {results_y1['regression_quality']}")
    print(f"  Classification Quality: {results_y1['classification_quality']}")
    print(f"  Stable Genes: {results_y1['high_stable_genes']}")
    
    print(f"\\nY2 (Live Birth):")
    print(f"  Best Model: {results_y2['best_model']}")
    print(f"  R² Performance: {results_y2['r2_performance']:.4f} ± {results_y2['r2_stability']:.4f}")
    print(f"  Classification Acc: {results_y2['classification_performance']:.4f}")
    print(f"  Regression Quality: {results_y2['regression_quality']}")
    print(f"  Classification Quality: {results_y2['classification_quality']}")
    print(f"  Stable Genes: {results_y2['high_stable_genes']}")
    
    if results_y1['r2_performance'] > results_y2['r2_performance']:
        print(f"\\nCLINICAL PREGNANCY (Y1) shows better regression performance.")
    elif results_y2['r2_performance'] > results_y1['r2_performance']:
        print(f"\\nLIVE BIRTH (Y2) shows better regression performance.")
    else:
        print(f"\\nBoth outcomes show similar regression performance.")
    
    print(f"\\nLinear Regression advantages:")
    print(f"  - Direct coefficient interpretation")
    print(f"  - Fast training and prediction")
    print(f"  - Multiple regularization options")
    print(f"  - Both regression and classification metrics")
    print(f"  - Good baseline for linear relationships")
    
    print(f"\\nAll analyses completed successfully!")
    print(f"Timestamp: {results_y1['timestamp']} (Y1), {results_y2['timestamp']} (Y2)")