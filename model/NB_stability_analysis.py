#!/usr/bin/env python3
"""
Naive Bayes Stability Analysis
Runs comprehensive stability analysis for both Y1 and Y2 outcomes
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from scipy.stats import pearsonr
from collections import Counter, defaultdict
import warnings
import pickle
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

def run_stability_analysis(selected_outcome='Y1'):
    """Run stability analysis for specified outcome"""
    print(f"\n{'='*60}")
    print(f"STARTING STABILITY ANALYSIS FOR {selected_outcome}")
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
        'test_indices': []
    }
    
    # Naive Bayes models to test
    models = {
        'Gaussian NB': GaussianNB(),
        'Multinomial NB': MultinomialNB(),
        'Bernoulli NB': BernoulliNB()
    }
    
    print(f"\nStarting stability testing with {N_ITERATIONS} iterations...")
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
        
        # Test each model
        for model_name, model in models.items():
            # Clone the model for this iteration
            model_copy = type(model)()
            
            # Choose appropriate scaling
            if model_name == 'Gaussian NB':
                scaler = StandardScaler()
                X_train_use = scaler.fit_transform(X_train_selected)
                X_test_use = scaler.transform(X_test_selected)
            else:
                scaler = MinMaxScaler()
                X_train_use = scaler.fit_transform(X_train_selected)
                X_test_use = scaler.transform(X_test_selected)
            
            # Train model
            model_copy.fit(X_train_use, y_train)
            
            # Make predictions
            y_pred = model_copy.predict(X_test_use)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC-ROC for binary classification
            if len(np.unique(y)) == 2 and hasattr(model_copy, 'predict_proba'):
                y_proba = model_copy.predict_proba(X_test_use)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = None
            
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
        'auc_roc': stability_results['auc_roc']
    })
    
    print("\nPerformance Summary by Model:")
    print("=" * 50)
    
    # Calculate summary statistics for each model
    summary_stats = results_df.groupby('model_name').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'auc_roc': ['mean', 'std']
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
    
    # Generate and save visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories if they don't exist
    result_num_dir = f'result_num/naive_bayes'
    result_pic_dir = f'result_pic/naive_bayes'
    os.makedirs(result_num_dir, exist_ok=True)
    os.makedirs(result_pic_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # 1. Performance comparison plot
    models_list = results_df['model_name'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy boxplot
    accuracy_data = [results_df[results_df['model_name'] == model]['accuracy'].values for model in models_list]
    axes[0, 0].boxplot(accuracy_data, labels=models_list)
    axes[0, 0].set_title(f'Accuracy Distribution - {SELECTED_OUTCOME}')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Performance metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models_list):
        model_data = results_df[results_df['model_name'] == model]
        means = [model_data[metric].mean() for metric in metrics]
        stds = [model_data[metric].std() for metric in metrics]
        axes[0, 1].bar(x + i*width, means, width, yerr=stds, label=model, alpha=0.8)
    
    axes[0, 1].set_title(f'Performance Metrics - {SELECTED_OUTCOME}')
    axes[0, 1].set_xlabel('Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(metrics, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature stability histogram
    axes[1, 0].hist(feature_stability_df['percentage'], bins=15, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(80, color='red', linestyle='--', label='80% threshold')
    axes[1, 0].axvline(50, color='orange', linestyle='--', label='50% threshold')
    axes[1, 0].set_xlabel('Selection Percentage')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title(f'Feature Selection Stability - {SELECTED_OUTCOME}')
    axes[1, 0].legend()
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
    plt.savefig(f'{result_pic_dir}/stability_analysis_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model stability comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    cv_data = []
    model_names = []
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        cv = model_data['accuracy'].std() / model_data['accuracy'].mean()
        cv_data.append(cv)
        model_names.append(model)
    
    bars = ax.bar(model_names, cv_data, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_title(f'Model Stability Comparison - {SELECTED_OUTCOME}\\n(Lower = More Stable)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Coefficient of Variation')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cv in zip(bars, cv_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{cv:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/model_stability_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature selection heatmap for top features
    if N_ITERATIONS >= 10:
        top_30_features = feature_stability_df.head(30)['feature'].tolist()
        heatmap_data = np.zeros((len(top_30_features), N_ITERATIONS))
        
        for i, iteration_features in enumerate(stability_results['selected_features'][:N_ITERATIONS]):
            for j, feature in enumerate(top_30_features):
                if feature in iteration_features:
                    heatmap_data[j, i] = 1
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='RdYlBu_r', xticklabels=range(1, N_ITERATIONS+1), 
                   yticklabels=top_30_features, ax=ax)
        ax.set_title(f'Feature Selection Pattern - {SELECTED_OUTCOME}\\n(Top 30 Features)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(f'{result_pic_dir}/feature_heatmap_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
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
    
    # Save complete stability data
    stability_data_filename = f'{result_num_dir}/complete_stability_data_{SELECTED_OUTCOME}_{timestamp}.pkl'
    with open(stability_data_filename, 'wb') as f:
        pickle.dump({
            'config': {
                'outcome': SELECTED_OUTCOME,
                'n_features': N_FEATURES,
                'n_iterations': N_ITERATIONS,
                'test_size': TEST_SIZE
            },
            'results': stability_results,
            'summary': {
                'best_model': best_overall,
                'model_rankings': combined_scores,
                'feature_stability': feature_stability_df,
                'performance_summary': summary_stats
            }
        }, f)
    
    print(f"Complete stability data saved to: {stability_data_filename}")
    
    # Save summary report
    report_filename = f'{result_num_dir}/stability_report_{SELECTED_OUTCOME}_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(f"NAIVE BAYES STABILITY ANALYSIS REPORT\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Outcome: {SELECTED_OUTCOME}\\n")
        f.write(f"Iterations: {N_ITERATIONS}\\n\\n")
        
        f.write(f"BEST MODEL: {best_overall}\\n")
        f.write(f"Performance: {best_model_data['accuracy'].mean():.4f} ± {best_model_data['accuracy'].std():.4f}\\n\\n")
        
        f.write(f"TOP 10 STABLE BIOMARKERS:\\n")
        for i, (_, row) in enumerate(feature_stability_df.head(10).iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:15} ({row['percentage']:5.1f}%)\\n")
        
        f.write(f"\\nSTABILITY METRICS:\\n")
        f.write(f"High stability genes (≥80%): {high_stable}\\n")
        f.write(f"Overall performance: {reliability}\\n")
    
    print(f"Summary report saved to: {report_filename}")
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS FOR {SELECTED_OUTCOME} COMPLETED SUCCESSFULLY!")
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
        'timestamp': timestamp
    }

if __name__ == "__main__":
    print("NAIVE BAYES STABILITY ANALYSIS")
    print("Running analysis for both Y1 and Y2 outcomes...")
    
    # Run for Y1 (Clinical pregnancy outcome)
    results_y1 = run_stability_analysis('Y1')
    
    # Run for Y2 (Live birth outcome)  
    results_y2 = run_stability_analysis('Y2')
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Y1 (Clinical Pregnancy):")
    print(f"  Best Model: {results_y1['best_model']}")
    print(f"  Performance: {results_y1['performance']:.4f} ± {results_y1['stability']:.4f}")
    print(f"  Reliability: {results_y1['reliability']}")
    print(f"  Stable Genes: {results_y1['high_stable_genes']}")
    
    print(f"\\nY2 (Live Birth):")
    print(f"  Best Model: {results_y2['best_model']}")
    print(f"  Performance: {results_y2['performance']:.4f} ± {results_y2['stability']:.4f}")
    print(f"  Reliability: {results_y2['reliability']}")
    print(f"  Stable Genes: {results_y2['high_stable_genes']}")
    
    if results_y1['performance'] > results_y2['performance']:
        print(f"\\nCLINICAL PREGNANCY (Y1) shows better predictive performance.")
    elif results_y2['performance'] > results_y1['performance']:
        print(f"\\nLIVE BIRTH (Y2) shows better predictive performance.")
    else:
        print(f"\\nBoth outcomes show similar predictive performance.")
    
    print(f"\\nAll analyses completed successfully!")
    print(f"Timestamp: {results_y1['timestamp']} (Y1), {results_y2['timestamp']} (Y2)")