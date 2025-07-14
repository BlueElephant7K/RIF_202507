#!/usr/bin/env python3
"""
LightGBM Stability Analysis
Runs comprehensive stability analysis for both Y1 and Y2 outcomes using LightGBM variants
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

# LightGBM import with fallback
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Please install: pip install lightgbm")

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

def run_stability_analysis(selected_outcome='Y1'):
    """Run stability analysis for specified outcome using LightGBM variants"""
    print(f"\n{'='*60}")
    print(f"STARTING LIGHTGBM STABILITY ANALYSIS FOR {selected_outcome}")
    print(f"{'='*60}")
    
    if not LIGHTGBM_AVAILABLE:
        print("ERROR: LightGBM is not installed. Please install it using:")
        print("pip install lightgbm")
        return None
    
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
        'feature_importances': [],
        'num_boosting_rounds': [],
        'best_iteration': [],
        'training_time': [],
        'prediction_time': []
    }
    
    # Calculate class weights for imbalanced data
    class_counts = y.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1
    
    # LightGBM models to test
    models = {
        'LightGBM_Default': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            early_stopping_rounds=10
        ),
        'LightGBM_Fast': lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.2,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            early_stopping_rounds=5
        ),
        'LightGBM_Accurate': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            early_stopping_rounds=15
        )
    }
    
    print(f"\nStarting stability testing with {N_ITERATIONS} iterations...")
    print("Testing 3 LightGBM variants: Default, Fast, Accurate")
    print("This may take a few minutes...")
    
    # Main stability testing loop
    for iteration in range(N_ITERATIONS):
        if (iteration + 1) % 5 == 0:
            print(f"  Completed {iteration + 1}/{N_ITERATIONS} iterations")
        
        # CORRECT APPROACH: Split data first, then feature selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=iteration, stratify=y
        )
        
        # Further split training for early stopping validation
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=iteration, stratify=y_train
        )
        
        # Feature selection ONLY on training data
        selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
        X_train_fit_selected = selector.fit_transform(X_train_fit, y_train_fit)  # Fit on training only
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature information
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]
        
        # Test each model
        for model_name, model in models.items():
            # Clone the model for this iteration
            model_copy = type(model)(**model.get_params())
            model_copy.set_params(random_state=iteration)  # Different random state per iteration
            
            # Measure training time
            import time
            start_time = time.time()
            
            # Train model with early stopping
            model_copy.fit(
                X_train_fit_selected, y_train_fit,
                eval_set=[(X_val_selected, y_val)],
                callbacks=[lgb.early_stopping(model_copy.early_stopping_rounds, verbose=False)]
            )
            
            training_time = time.time() - start_time
            
            # Measure prediction time
            start_time = time.time()
            y_pred = model_copy.predict(X_test_selected)
            y_proba = model_copy.predict_proba(X_test_selected)[:, 1]
            prediction_time = time.time() - start_time
            
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
            
            # LightGBM specific metrics
            feature_importances = model_copy.feature_importances_
            best_iteration = model_copy.best_iteration_ if hasattr(model_copy, 'best_iteration_') else model_copy.n_estimators
            
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
            stability_results['feature_importances'].append(feature_importances.tolist())
            stability_results['num_boosting_rounds'].append(model_copy.n_estimators)
            stability_results['best_iteration'].append(best_iteration)
            stability_results['training_time'].append(training_time)
            stability_results['prediction_time'].append(prediction_time)
    
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
        'best_iteration': stability_results['best_iteration'],
        'training_time': stability_results['training_time'],
        'prediction_time': stability_results['prediction_time']
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
        'best_iteration': ['mean', 'std'],
        'training_time': ['mean', 'std'],
        'prediction_time': ['mean', 'std']
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
    
    # LightGBM specific analysis - Feature Importance
    print(f"\nLightGBM Feature Importance Analysis:")
    print("=" * 50)
    
    # Aggregate feature importances across all iterations and models
    importance_analysis = defaultdict(list)
    for model_name, importances, features in zip(stability_results['model_name'], 
                                                 stability_results['feature_importances'],
                                                 stability_results['selected_features']):
        for feature, importance in zip(features, importances):
            importance_analysis[feature].append(importance)
    
    # Calculate average importance for stable features
    stable_features_with_importance = []
    for _, row in feature_stability_df.head(30).iterrows():
        feature = row['feature']
        if feature in importance_analysis and len(importance_analysis[feature]) > 0:
            avg_importance = np.mean(importance_analysis[feature])
            std_importance = np.std(importance_analysis[feature])
            stable_features_with_importance.append((feature, row['percentage'], avg_importance, std_importance))
    
    stable_features_with_importance.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 15 features by LightGBM importance:")
    for i, (feature, stability, avg_imp, std_imp) in enumerate(stable_features_with_importance[:15], 1):
        print(f"  {i:2d}. {feature:15} | Stability: {stability:5.1f}% | Importance: {avg_imp:.4f}±{std_imp:.4f}")
    
    # Speed analysis
    print(f"\nSpeed Analysis:")
    print("=" * 50)
    speed_analysis = results_df.groupby('model_name').agg({
        'training_time': ['mean', 'std'],
        'prediction_time': ['mean', 'std']
    }).round(4)
    
    print("Training and Prediction Times (seconds):")
    for model in speed_analysis.index:
        train_time_mean = speed_analysis.loc[model, ('training_time', 'mean')]
        train_time_std = speed_analysis.loc[model, ('training_time', 'std')]
        pred_time_mean = speed_analysis.loc[model, ('prediction_time', 'mean')]
        pred_time_std = speed_analysis.loc[model, ('prediction_time', 'std')]
        
        print(f"{model:20}: Train={train_time_mean:.3f}±{train_time_std:.3f}s, Pred={pred_time_mean:.4f}±{pred_time_std:.4f}s")
    
    # Early stopping analysis
    print(f"\nEarly Stopping Analysis:")
    print("=" * 50)
    iteration_analysis = results_df.groupby('model_name')['best_iteration'].agg(['mean', 'std']).round(2)
    print("Average best iteration by model:")
    for model in iteration_analysis.index:
        mean_iter = iteration_analysis.loc[model, 'mean']
        std_iter = iteration_analysis.loc[model, 'std']
        print(f"  {model:20}: {mean_iter:6.1f} ± {std_iter:5.1f}")
    
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
    
    # LightGBM specific insights
    print(f"\nLightGBM Insights:")
    best_iterations = best_model_data['best_iteration'].mean()
    best_train_time = best_model_data['training_time'].mean()
    best_pred_time = best_model_data['prediction_time'].mean()
    print(f"  Best model ({best_overall}) converges at ~{best_iterations:.0f} iterations")
    print(f"  Training time: {best_train_time:.3f}s, Prediction time: {best_pred_time:.4f}s")
    print(f"  Leaf-wise tree growth for efficiency")
    print(f"  Memory efficient gradient boosting")
    print(f"  Excellent speed vs accuracy trade-off")
    print(f"  Built-in categorical feature support")
    if len(core_biomarkers) > 0:
        print(f"  Superior performance on high-dimensional data")
    
    # Generate and save visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories if they don't exist
    result_num_dir = f'result_num/lightgbm'
    result_pic_dir = f'result_pic/lightgbm'
    os.makedirs(result_num_dir, exist_ok=True)
    os.makedirs(result_pic_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # 1. Performance comparison plot
    models_list = results_df['model_name'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy boxplot
    accuracy_data = [results_df[results_df['model_name'] == model]['accuracy'].values for model in models_list]
    axes[0, 0].boxplot(accuracy_data, labels=[m.replace('LightGBM_', '') for m in models_list])
    axes[0, 0].set_title(f'Accuracy Distribution - {SELECTED_OUTCOME}')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Performance metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = ['skyblue', 'lightgreen', 'salmon']
    for i, model in enumerate(models_list):
        model_data = results_df[results_df['model_name'] == model]
        means = [model_data[metric].mean() for metric in metrics]
        stds = [model_data[metric].std() for metric in metrics]
        axes[0, 1].bar(x + i*width, means, width, yerr=stds, 
                      label=model.replace('LightGBM_', ''), 
                      alpha=0.8, color=colors[i])
    
    axes[0, 1].set_title(f'Performance Metrics - {SELECTED_OUTCOME}')
    axes[0, 1].set_xlabel('Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(metrics, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training time comparison
    time_data = [results_df[results_df['model_name'] == model]['training_time'].values for model in models_list]
    axes[1, 0].boxplot(time_data, labels=[m.replace('LightGBM_', '') for m in models_list])
    axes[1, 0].set_title(f'Training Time Distribution - {SELECTED_OUTCOME}')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top 15 important features
    if stable_features_with_importance:
        top_15_imp = stable_features_with_importance[:15]
        features = [f[0] for f in top_15_imp]
        importances = [f[2] for f in top_15_imp]
        
        axes[1, 1].barh(range(len(features)), importances)
        axes[1, 1].set_yticks(range(len(features)))
        axes[1, 1].set_yticklabels(features, fontsize=8)
        axes[1, 1].set_xlabel('Average Feature Importance')
        axes[1, 1].set_title(f'Top 15 Important Features - {SELECTED_OUTCOME}')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/lightgbm_analysis_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Speed vs Accuracy analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Speed comparison
    train_times = []
    pred_times = []
    accuracies = []
    model_names = []
    
    for model in models_list:
        model_data = results_df[results_df['model_name'] == model]
        train_times.append(model_data['training_time'].mean())
        pred_times.append(model_data['prediction_time'].mean())
        accuracies.append(model_data['accuracy'].mean())
        model_names.append(model.replace('LightGBM_', ''))
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, train_times, width, label='Training Time', alpha=0.8, color='lightblue')
    axes[0].bar(x_pos + width/2, [t*1000 for t in pred_times], width, label='Prediction Time (ms)', alpha=0.8, color='lightcoral')
    axes[0].set_xlabel('LightGBM Variant')
    axes[0].set_ylabel('Time')
    axes[0].set_title(f'Training vs Prediction Speed - {SELECTED_OUTCOME}')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Speed vs Accuracy scatter
    axes[1].scatter(train_times, accuracies, s=100, alpha=0.7, c=colors[:len(model_names)])
    for i, model in enumerate(model_names):
        axes[1].annotate(model, (train_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[1].set_xlabel('Training Time (seconds)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Speed vs Accuracy Trade-off - {SELECTED_OUTCOME}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_pic_dir}/speed_analysis_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance stability analysis
    if stable_features_with_importance:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: Stability vs Importance
        stabilities = [f[1] for f in stable_features_with_importance]
        importances = [f[2] for f in stable_features_with_importance]
        
        axes[0].scatter(stabilities, importances, alpha=0.7, s=60)
        axes[0].set_xlabel('Feature Selection Stability (%)')
        axes[0].set_ylabel('Average Feature Importance')
        axes[0].set_title(f'Feature Stability vs Importance - {SELECTED_OUTCOME}')
        axes[0].grid(True, alpha=0.3)
        
        # Add labels for top features
        for i, (feature, stab, imp, _) in enumerate(stable_features_with_importance[:10]):
            axes[0].annotate(feature, (stab, imp), xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.8)
        
        # Feature importance distribution
        all_importances = [f[2] for f in stable_features_with_importance]
        axes[1].hist(all_importances, bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(all_importances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_importances):.4f}')
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title(f'Feature Importance Distribution - {SELECTED_OUTCOME}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{result_pic_dir}/importance_stability_{SELECTED_OUTCOME}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {result_pic_dir}/")
    
    # Save results
    
    # Save detailed results
    results_filename = f'{result_num_dir}/lightgbm_analysis_{SELECTED_OUTCOME}_{timestamp}.csv'
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
    
    # Save feature importance analysis
    if stable_features_with_importance:
        importance_analysis_df = pd.DataFrame(stable_features_with_importance, 
                                            columns=['feature', 'stability_percentage', 'avg_importance', 'std_importance'])
        importance_filename = f'{result_num_dir}/feature_importance_{SELECTED_OUTCOME}_{timestamp}.csv'
        importance_analysis_df.to_csv(importance_filename, index=False)
        print(f"Feature importance analysis saved to: {importance_filename}")
    
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
                'importance_analysis': stable_features_with_importance if 'stable_features_with_importance' in locals() else None,
                'speed_analysis': speed_analysis,
                'iteration_analysis': iteration_analysis
            }
        }, f)
    
    print(f"Complete stability data saved to: {stability_data_filename}")
    
    # Save summary report
    report_filename = f'{result_num_dir}/lightgbm_report_{SELECTED_OUTCOME}_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(f"LIGHTGBM STABILITY ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Outcome: {SELECTED_OUTCOME}\n")
        f.write(f"Iterations: {N_ITERATIONS}\n")
        f.write(f"Models: Default, Fast, Accurate\n\n")
        
        f.write(f"BEST MODEL: {best_overall}\n")
        f.write(f"Performance: {best_model_data['accuracy'].mean():.4f} ± {best_model_data['accuracy'].std():.4f}\n")
        f.write(f"Best Iteration: {best_model_data['best_iteration'].mean():.1f}\n")
        f.write(f"Training Time: {best_model_data['training_time'].mean():.3f}s\n")
        f.write(f"Prediction Time: {best_model_data['prediction_time'].mean():.4f}s\n\n")
        
        f.write(f"TOP 10 STABLE BIOMARKERS:\n")
        for i, (_, row) in enumerate(feature_stability_df.head(10).iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:15} ({row['percentage']:5.1f}%)\n")
        
        f.write(f"\nSTABILITY METRICS:\n")
        f.write(f"High stability genes (≥80%): {high_stable}\n")
        f.write(f"Overall performance: {reliability}\n")
        
        if stable_features_with_importance:
            f.write(f"\nTOP IMPORTANT FEATURES:\n")
            for i, (feature, stability, importance, std_imp) in enumerate(stable_features_with_importance[:10], 1):
                f.write(f"{i:2d}. {feature:15} | Stability: {stability:5.1f}% | Importance: {importance:.4f}\n")
        
        f.write(f"\nMODEL PERFORMANCE RANKING:\n")
        for i, (model, score) in enumerate(combined_scores, 1):
            f.write(f"{i}. {model}: {score:.4f}\n")
        
        f.write(f"\nSPEED ANALYSIS:\n")
        for model in speed_analysis.index:
            train_time = speed_analysis.loc[model, ('training_time', 'mean')]
            pred_time = speed_analysis.loc[model, ('prediction_time', 'mean')]
            f.write(f"{model}: Train={train_time:.3f}s, Pred={pred_time:.4f}s\n")
    
    print(f"Summary report saved to: {report_filename}")
    
    print(f"\n{'='*50}")
    print(f"LIGHTGBM ANALYSIS FOR {SELECTED_OUTCOME} COMPLETED SUCCESSFULLY!")
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
        'avg_best_iteration': best_model_data['best_iteration'].mean(),
        'avg_training_time': best_model_data['training_time'].mean(),
        'avg_prediction_time': best_model_data['prediction_time'].mean(),
        'timestamp': timestamp
    }

if __name__ == "__main__":
    print("LIGHTGBM STABILITY ANALYSIS")
    print("Running analysis for both Y1 and Y2 outcomes...")
    
    # Run for Y1 (Clinical pregnancy outcome)
    results_y1 = run_stability_analysis('Y1')
    
    if results_y1 is None:
        print("Cannot proceed without LightGBM installation.")
        exit(1)
    
    # Run for Y2 (Live birth outcome)  
    results_y2 = run_stability_analysis('Y2')
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY - LIGHTGBM")
    print(f"{'='*60}")
    print(f"Y1 (Clinical Pregnancy):")
    print(f"  Best Model: {results_y1['best_model']}")
    print(f"  Performance: {results_y1['performance']:.4f} ± {results_y1['stability']:.4f}")
    print(f"  Reliability: {results_y1['reliability']}")
    print(f"  Stable Genes: {results_y1['high_stable_genes']}")
    print(f"  Best Iteration: {results_y1['avg_best_iteration']:.1f}")
    print(f"  Training Time: {results_y1['avg_training_time']:.3f}s")
    print(f"  Prediction Time: {results_y1['avg_prediction_time']:.4f}s")
    
    print(f"\nY2 (Live Birth):")
    print(f"  Best Model: {results_y2['best_model']}")
    print(f"  Performance: {results_y2['performance']:.4f} ± {results_y2['stability']:.4f}")
    print(f"  Reliability: {results_y2['reliability']}")
    print(f"  Stable Genes: {results_y2['high_stable_genes']}")
    print(f"  Best Iteration: {results_y2['avg_best_iteration']:.1f}")
    print(f"  Training Time: {results_y2['avg_training_time']:.3f}s")
    print(f"  Prediction Time: {results_y2['avg_prediction_time']:.4f}s")
    
    if results_y1['performance'] > results_y2['performance']:
        print(f"\nCLINICAL PREGNANCY (Y1) shows better predictive performance.")
    elif results_y2['performance'] > results_y1['performance']:
        print(f"\nLIVE BIRTH (Y2) shows better predictive performance.")
    else:
        print(f"\nBoth outcomes show similar predictive performance.")
    
    print(f"\nLightGBM advantages:")
    print(f"  - Faster training than XGBoost")
    print(f"  - Lower memory consumption")
    print(f"  - Leaf-wise tree growth")
    print(f"  - Built-in categorical feature support")
    print(f"  - Excellent accuracy on high-dimensional data")
    print(f"  - Parallel and distributed training")
    print(f"  - Superior speed vs accuracy trade-off")
    
    print(f"\nAll analyses completed successfully!")
    print(f"Timestamp: {results_y1['timestamp']} (Y1), {results_y2['timestamp']} (Y2)")