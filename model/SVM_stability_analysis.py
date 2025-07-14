#!/usr/bin/env python3
"""
SVM Stability Analysis
Runs comprehensive stability analysis for both Y1 and Y2 outcomes using SVM variants
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
from sklearn.svm import SVC, NuSVC
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
    """Run stability analysis for specified outcome using SVM variants"""
    print(f"\n{'='*60}")
    print(f"STARTING SVM STABILITY ANALYSIS FOR {selected_outcome}")
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
    y = df[SELECTED_OUTCOME].copy()
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
    
    # Initialize storage for stability results
    stability_results = {
        'iteration': [],
        'model_name': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc_roc': [],
        'selected_features': [],
        'feature_scores': [],
        'predictions': [],
        'test_indices': [],
        'n_support_vectors': [],
        'decision_function_values': [],
        'feature_coefficients': []
    }
    
    # SVM models to test with different kernels
    models = {
        'SVM_Linear': SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42),
        'SVM_Poly': SVC(kernel='poly', degree=3, C=1.0, class_weight='balanced', probability=True, random_state=42),
        'Nu_SVM': NuSVC(kernel='rbf', nu=0.5, class_weight='balanced', probability=True, random_state=42)
    }
    
    print(f"\nStarting stability testing with {N_ITERATIONS} iterations...")
    print("Testing 4 SVM variants: Linear, RBF, Polynomial, Nu-SVM")
    print("This may take a few minutes...")
    
    # Main stability testing loop
    for iteration in range(N_ITERATIONS):
        if (iteration + 1) % 5 == 0:
            print(f"  Completed {iteration + 1}/{N_ITERATIONS} iterations")
        
        # CORRECT APPROACH: Split data first, then feature selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=iteration, stratify=y
        )
        
        # Feature selection ONLY on training data
        selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
        X_train_selected = selector.fit_transform(X_train, y_train)  # Fit on training only
        X_test_selected = selector.transform(X_test)  # Transform test set
        
        # Get selected feature information
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]
        
        # Scale features (CRITICAL for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Test each model
        for model_name, model in models.items():
            # Clone the model for this iteration
            model_copy = type(model)(**model.get_params())
            
            # Train model
            model_copy.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model_copy.predict(X_test_scaled)
            y_proba = model_copy.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC-ROC for binary classification
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = None
            
            # SVM-specific metrics
            n_support_vectors = model_copy.n_support_.sum()
            decision_function_values = model_copy.decision_function(X_test_scaled)
            
            # Get feature coefficients for Linear SVM
            if model_name == 'SVM_Linear' and hasattr(model_copy, 'coef_'):
                feature_coefficients = model_copy.coef_[0]
            else:
                feature_coefficients = None
            
            # Store results
            stability_results['iteration'].append(iteration)
            stability_results['model_name'].append(model_name)
            stability_results['accuracy'].append(accuracy)
            stability_results['precision'].append(precision)
            stability_results['recall'].append(recall)
            stability_results['f1_score'].append(f1)
            stability_results['auc_roc'].append(auc)
            stability_results['selected_features'].append(list(selected_features))
            stability_results['feature_scores'].append(list(feature_scores))
            stability_results['predictions'].append(list(y_pred))
            stability_results['test_indices'].append(list(X_test.index))
            stability_results['n_support_vectors'].append(n_support_vectors)
            stability_results['decision_function_values'].append(list(decision_function_values))
            stability_results['feature_coefficients'].append(feature_coefficients.tolist() if feature_coefficients is not None else None)
    
    print(f"\nStability testing completed!")
    print(f"Total experiments: {len(stability_results['iteration'])}")
    
    # Process results
    results_df = pd.DataFrame({
        'iteration': stability_results['iteration'],
        'model_name': stability_results['model_name'],
        'accuracy': stability_results['accuracy'],
        'precision': stability_results['precision'],
        'recall': stability_results['recall'],
        'f1_score': stability_results['f1_score'],
        'auc_roc': stability_results['auc_roc'],
        'n_support_vectors': stability_results['n_support_vectors']
    })
    
    print("\nPerformance Summary by Model:")
    print("=" * 50)
    
    # Calculate summary statistics for each model
    summary_stats = results_df.groupby('model_name').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'auc_roc': ['mean', 'std'],
        'n_support_vectors': ['mean', 'std']
    }).round(4)
    
    print(summary_stats)
    
    # Best performing model on average
    best_model = results_df.groupby('model_name')['accuracy'].mean().idxmax()
    best_accuracy = results_df.groupby('model_name')['accuracy'].mean().max()
    
    print(f"\nBest performing model: {best_model}")
    print(f"Average accuracy: {best_accuracy:.4f}")
    
    # Feature stability analysis
    all_features = set()
    feature_frequency = Counter()
    
    # Count how often each feature is selected across ALL model iterations
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
    
    # Create feature stability dataframe with CORRECT percentage calculation
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
    
    # Core biomarkers (selected in ≥80% of iterations)
    core_biomarkers = feature_stability_df[feature_stability_df['percentage'] >= 80]
    print(f"\nCore Biomarkers (≥80% selection rate): {len(core_biomarkers)} genes")
    if len(core_biomarkers) > 0:
        print("Core genes:", list(core_biomarkers['feature'].head(10)))
    
    # SVM-specific analysis
    print(f"\nSVM-Specific Analysis:")
    print("=" * 50)
    
    # Support vector analysis
    sv_analysis = results_df.groupby('model_name')['n_support_vectors'].agg(['mean', 'std']).round(2)
    print("Support Vectors by Model:")
    for model in sv_analysis.index:
        mean_sv = sv_analysis.loc[model, 'mean']
        std_sv = sv_analysis.loc[model, 'std']
        print(f"  {model:12}: {mean_sv:6.1f} ± {std_sv:5.1f}")
    
    # Linear SVM coefficient analysis
    print(f"\nLinear SVM Feature Coefficient Analysis:")
    print("=" * 50)
    linear_results = [(i, coefs, features) for i, (model, coefs, features) in 
                      enumerate(zip(stability_results['model_name'], 
                                   stability_results['feature_coefficients'],
                                   stability_results['selected_features'])) if model == 'SVM_Linear' and coefs is not None]
    
    if linear_results:
        # Average coefficient magnitudes for stable features
        coefficient_analysis = defaultdict(list)
        for iteration, coefficients, features in linear_results:
            for feature, coef in zip(features, coefficients):
                coefficient_analysis[feature].append(abs(coef))
        
        # Calculate average coefficient magnitude for top stable features
        stable_features_with_coefs = []
        for _, row in feature_stability_df.head(20).iterrows():
            feature = row['feature']
            if feature in coefficient_analysis and len(coefficient_analysis[feature]) > 0:
                avg_coef_magnitude = np.mean(coefficient_analysis[feature])
                stable_features_with_coefs.append((feature, row['percentage'], avg_coef_magnitude))
        
        stable_features_with_coefs.sort(key=lambda x: x[2], reverse=True)
        
        print("Top 10 features by coefficient magnitude (Linear SVM):")
        for i, (feature, stability, coef_mag) in enumerate(stable_features_with_coefs[:10], 1):
            print(f"  {i:2d}. {feature:15} | Stability: {stability:5.1f}% | Avg |coef|: {coef_mag:.4f}")
    
    # Kernel comparison
    print(f"\nKernel Performance Comparison:")
    print("=" * 50)
    kernel_comparison = results_df.groupby('model_name').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).round(4)
    
    for model in kernel_comparison.index:
        acc_mean = kernel_comparison.loc[model, ('accuracy', 'mean')]
        acc_std = kernel_comparison.loc[model, ('accuracy', 'std')]
        f1_mean = kernel_comparison.loc[model, ('f1_score', 'mean')]
        f1_std = kernel_comparison.loc[model, ('f1_score', 'std')]
        print(f"{model:12}: Acc={acc_mean:.4f}±{acc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}")
    
    # Model ranking analysis
    models_list = results_df['model_name'].unique()
    
    # Combined score (accuracy - stability penalty)
    combined_scores = []
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        mean_acc = model_data['accuracy'].mean()
        cv = model_data['accuracy'].std() / mean_acc
        combined_score = mean_acc - cv  # Penalize instability
        combined_scores.append((model, combined_score))
    
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nModel Ranking by Combined Score (Accuracy - Stability Penalty):")
    for i, (model, score) in enumerate(combined_scores, 1):
        print(f"  {i}. {model}: {score:.4f}")
    
    # Generate comprehensive summary
    best_overall = combined_scores[0][0]
    best_model_data = results_df[results_df['model_name'] == best_overall]
    overall_mean = results_df['accuracy'].mean()
    
    if overall_mean > 0.8:
        reliability = "Excellent"
    elif overall_mean > 0.7:
        reliability = "Good"
    elif overall_mean > 0.6:
        reliability = "Moderate"
    else:
        reliability = "Poor"
    
    high_stable = len(feature_stability_df[feature_stability_df['percentage'] >= 80])
    
    print(f"\nCOMPREHENSIVE SUMMARY FOR {SELECTED_OUTCOME}:")
    print("=" * 50)
    print(f"Best Overall Model: {best_overall}")
    print(f"Performance: {best_model_data['accuracy'].mean():.4f} ± {best_model_data['accuracy'].std():.4f}")
    print(f"Overall Performance Level: {reliability}")
    print(f"Highly Stable Genes (≥80%): {high_stable}")
    print(f"Top Biomarker: {feature_stability_df.iloc[0]['feature']} ({feature_stability_df.iloc[0]['percentage']:.1f}%)")
    
    # SVM specific insights
    print(f"\nSVM Insights:")
    best_sv_count = results_df[results_df['model_name'] == best_overall]['n_support_vectors'].mean()
    print(f"  Best model ({best_overall}) uses ~{best_sv_count:.0f} support vectors on average")
    print(f"  Linear SVM: Provides interpretable feature weights")
    print(f"  RBF SVM: Captures non-linear gene relationships")
    print(f"  Polynomial SVM: Models feature interactions")
    print(f"  Nu-SVM: Alternative formulation with nu parameter")
    if len(core_biomarkers) > 0:
        print(f"  Support vectors likely identify critical samples for classification")
    
    # Generate and save visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories if they don't exist
    result_num_dir = f'result_num/svm'
    result_pic_dir = f'result_pic/svm'
    os.makedirs(result_num_dir, exist_ok=True)
    os.makedirs(result_pic_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # 1. Performance comparison plot
    models_list = results_df['model_name'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy boxplot
    accuracy_data = [results_df[results_df['model_name'] == model]['accuracy'].values for model in models_list]
    axes[0, 0].boxplot(accuracy_data, labels=[m.replace('SVM_', '').replace('Nu_SVM', 'Nu-SVM') for m in models_list])
    axes[0, 0].set_title(f'Accuracy Distribution - {SELECTED_OUTCOME}')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Performance metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.2
    
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange']
    for i, model in enumerate(models_list):
        model_data = results_df[results_df['model_name'] == model]
        means = [model_data[metric].mean() for metric in metrics]
        stds = [model_data[metric].std() for metric in metrics]
        axes[0, 1].bar(x + i*width, means, width, yerr=stds, 
                      label=model.replace('SVM_', '').replace('Nu_SVM', 'Nu-SVM'), 
                      alpha=0.8, color=colors[i])
    
    axes[0, 1].set_title(f'Performance Metrics - {SELECTED_OUTCOME}')
    axes[0, 1].set_xlabel('Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x + width*1.5)
    axes[0, 1].set_xticklabels(metrics, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Support vectors comparison
    sv_data = [results_df[results_df['model_name'] == model]['n_support_vectors'].values for model in models_list]
    axes[1, 0].boxplot(sv_data, labels=[m.replace('SVM_', '').replace('Nu_SVM', 'Nu-SVM') for m in models_list])
    axes[1, 0].set_title(f'Support Vector Count - {SELECTED_OUTCOME}')
    axes[1, 0].set_ylabel('Number of Support Vectors')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top 15 stable features
    top_15 = feature_stability_df.head(15)
    axes[1, 1].barh(range(len(top_15)), top_15['percentage'])
    axes[1, 1].set_yticks(range(len(top_15)))
    axes[1, 1].set_yticklabels(top_15['feature'], fontsize=8)
    axes[1, 1].set_xlabel('Selection Percentage')
    axes[1, 1].set_title(f'Top 15 Stable Features - {SELECTED_OUTCOME}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/svm_analysis_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Kernel performance and stability comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model stability comparison
    cv_data = []
    model_names = []
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        cv = model_data['accuracy'].std() / model_data['accuracy'].mean()
        cv_data.append(cv)
        model_names.append(model.replace('SVM_', '').replace('Nu_SVM', 'Nu-SVM'))
    
    bars = axes[0].bar(model_names, cv_data, alpha=0.8, color=colors)
    axes[0].set_title(f'SVM Kernel Stability - {SELECTED_OUTCOME}\n(Lower = More Stable)')
    axes[0].set_xlabel('SVM Variant')
    axes[0].set_ylabel('Coefficient of Variation')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cv in zip(bars, cv_data):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{cv:.4f}', ha='center', va='bottom')
    
    # AUC comparison if available
    if results_df['auc_roc'].notna().any():
        auc_data = []
        for model in models_list:
            model_data = results_df[results_df['model_name'] == model]
            auc_mean = model_data['auc_roc'].mean()
            auc_data.append(auc_mean)
        
        bars2 = axes[1].bar(model_names, auc_data, alpha=0.8, color=colors)
        axes[1].set_title(f'AUC-ROC Comparison - {SELECTED_OUTCOME}')
        axes[1].set_xlabel('SVM Variant')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, auc in zip(bars2, auc_data):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
    else:
        axes[1].text(0.5, 0.5, 'AUC-ROC not available\nfor multi-class', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('AUC-ROC Comparison')
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/kernel_comparison_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Linear SVM coefficient analysis
    if stable_features_with_coefs:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        top_coef_features = stable_features_with_coefs[:15]  # Top 15 by coefficient magnitude
        features = [f[0] for f in top_coef_features]
        coef_mags = [f[2] for f in top_coef_features]
        
        bars = ax.barh(range(len(features)), coef_mags, alpha=0.8, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Average |Coefficient| Magnitude')
        ax.set_title(f'Top Features by Linear SVM Coefficients - {SELECTED_OUTCOME}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, coef) in enumerate(zip(bars, coef_mags)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{coef:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{result_pic_dir}/linear_svm_coefficients_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {result_pic_dir}/")
    
    # Save results
    
    # Save detailed results
    results_filename = f'{result_num_dir}/svm_analysis_{SELECTED_OUTCOME}_{timestamp}.csv'
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
    
    # Save Linear SVM coefficient analysis
    if stable_features_with_coefs:
        coef_analysis_df = pd.DataFrame(stable_features_with_coefs, 
                                       columns=['feature', 'stability_percentage', 'avg_coefficient_magnitude'])
        coef_filename = f'{result_num_dir}/linear_svm_coefficients_{SELECTED_OUTCOME}_{timestamp}.csv'
        coef_analysis_df.to_csv(coef_filename, index=False)
        print(f"Linear SVM coefficient analysis saved to: {coef_filename}")
    
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
                'support_vector_analysis': sv_analysis,
                'coefficient_analysis': stable_features_with_coefs if 'stable_features_with_coefs' in locals() else None
            }
        }, f)
    
    print(f"Complete stability data saved to: {stability_data_filename}")
    
    # Save summary report
    report_filename = f'{result_num_dir}/svm_report_{SELECTED_OUTCOME}_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(f"SVM STABILITY ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Outcome: {SELECTED_OUTCOME}\n")
        f.write(f"Iterations: {N_ITERATIONS}\n")
        f.write(f"Models: Linear, RBF, Polynomial, Nu-SVM\n\n")
        
        f.write(f"BEST MODEL: {best_overall}\n")
        f.write(f"Performance: {best_model_data['accuracy'].mean():.4f} ± {best_model_data['accuracy'].std():.4f}\n")
        f.write(f"Support Vectors: ~{best_model_data['n_support_vectors'].mean():.0f}\n\n")
        
        f.write(f"TOP 10 STABLE BIOMARKERS:\n")
        for i, (_, row) in enumerate(feature_stability_df.head(10).iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:15} ({row['percentage']:5.1f}%)\n")
        
        f.write(f"\nSTABILITY METRICS:\n")
        f.write(f"High stability genes (≥80%): {high_stable}\n")
        f.write(f"Overall performance: {reliability}\n")
        
        if stable_features_with_coefs:
            f.write(f"\nTOP LINEAR SVM COEFFICIENT FEATURES:\n")
            for i, (feature, stability, coef_mag) in enumerate(stable_features_with_coefs[:10], 1):
                f.write(f"{i:2d}. {feature:15} | Stability: {stability:5.1f}% | |coef|: {coef_mag:.4f}\n")
        
        f.write(f"\nKERNEL PERFORMANCE RANKING:\n")
        for i, (model, score) in enumerate(combined_scores, 1):
            f.write(f"{i}. {model}: {score:.4f}\n")
    
    print(f"Summary report saved to: {report_filename}")
    
    print(f"\n{'='*50}")
    print(f"SVM ANALYSIS FOR {SELECTED_OUTCOME} COMPLETED SUCCESSFULLY!")
    print(f"Results directory: {result_num_dir}")
    print(f"Figures directory: {result_pic_dir}")
    print(f"Analysis timestamp: {timestamp}")
    print(f"{'='*50}")
    
    return {
        'outcome': SELECTED_OUTCOME,
        'best_model': best_overall,
        'performance': best_model_data['accuracy'].mean(),
        'stability': best_model_data['accuracy'].std(),
        'reliability': reliability,
        'high_stable_genes': high_stable,
        'avg_support_vectors': best_model_data['n_support_vectors'].mean(),
        'timestamp': timestamp
    }

if __name__ == "__main__":
    print("SVM STABILITY ANALYSIS")
    print("Running analysis for both Y1 and Y2 outcomes...")
    
    # Run for Y1 (Clinical pregnancy outcome)
    results_y1 = run_stability_analysis('Y1')
    
    # Run for Y2 (Live birth outcome)  
    results_y2 = run_stability_analysis('Y2')
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY - SVM")
    print(f"{'='*60}")
    print(f"Y1 (Clinical Pregnancy):")
    print(f"  Best Model: {results_y1['best_model']}")
    print(f"  Performance: {results_y1['performance']:.4f} ± {results_y1['stability']:.4f}")
    print(f"  Reliability: {results_y1['reliability']}")
    print(f"  Stable Genes: {results_y1['high_stable_genes']}")
    print(f"  Support Vectors: ~{results_y1['avg_support_vectors']:.0f}")
    
    print(f"\nY2 (Live Birth):")
    print(f"  Best Model: {results_y2['best_model']}")
    print(f"  Performance: {results_y2['performance']:.4f} ± {results_y2['stability']:.4f}")
    print(f"  Reliability: {results_y2['reliability']}")
    print(f"  Stable Genes: {results_y2['high_stable_genes']}")
    print(f"  Support Vectors: ~{results_y2['avg_support_vectors']:.0f}")
    
    if results_y1['performance'] > results_y2['performance']:
        print(f"\nCLINICAL PREGNANCY (Y1) shows better predictive performance.")
    elif results_y2['performance'] > results_y1['performance']:
        print(f"\nLIVE BIRTH (Y2) shows better predictive performance.")
    else:
        print(f"\nBoth outcomes show similar predictive performance.")
    
    print(f"\nSVM advantages:")
    print(f"  - Excellent for high-dimensional data")
    print(f"  - Multiple kernels for different data patterns")
    print(f"  - Strong theoretical foundation")
    print(f"  - Good generalization through maximum margin")
    print(f"  - Linear SVM provides interpretable coefficients")
    
    print(f"\nAll analyses completed successfully!")
    print(f"Timestamp: {results_y1['timestamp']} (Y1), {results_y2['timestamp']} (Y2)")