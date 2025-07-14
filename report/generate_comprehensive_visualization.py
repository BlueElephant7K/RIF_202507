import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
base_path = Path("/Users/nan/Documents/blueelephant/RIF_202507")
result_num_path = base_path / "result_num"
report_path = base_path / "report"

# Model configurations
models_config = {
    'Random Forest': {
        'path': 'random_forest',
        'file': 'rf_analysis_Y1_20250714_225648.csv',
        'name_col': 'model_name'
    },
    'XGBoost': {
        'path': 'xgboost',
        'file': 'xgb_analysis_Y1_20250714_230314.csv',
        'name_col': 'model_name'
    },
    'LightGBM': {
        'path': 'lightgbm',
        'file': 'lightgbm_analysis_Y1_20250714_230917.csv',
        'name_col': 'model_name'
    },
    'SVM': {
        'path': 'svm',
        'file': 'svm_analysis_Y1_20250714_225034.csv',
        'name_col': 'model_name'
    },
    'Logistic Regression': {
        'path': 'logistic_regression',
        'file': 'stability_analysis_Y1_20250714_224316.csv',
        'name_col': 'model_name'
    },
    'Linear Regression': {
        'path': 'linear_regression',
        'file': 'stability_analysis_Y1_20250714_224702.csv',
        'name_col': 'model_name'
    },
    'Naive Bayes': {
        'path': 'naive_bayes',
        'file': 'stability_analysis_Y1_20250714_224251.csv',
        'name_col': 'model_name'
    },
    'KNN': {
        'path': 'knn',
        'file': 'knn_analysis_Y1_20250714_231822.csv',
        'name_col': 'model_name'
    },
    'LDA/QDA': {
        'path': 'lda_qda',
        'file': 'lda_qda_analysis_Y1_20250714_231539.csv',
        'name_col': 'model_name'
    }
}

# Collect data from all models
all_model_data = []

for model_name, config in models_config.items():
    file_path = result_num_path / config['path'] / config['file']
    
    try:
        df = pd.read_csv(file_path)
        # Get top 10 iterations based on AUC
        top_10 = df.nlargest(10, 'auc_roc')
        
        # Add model category
        top_10['model_category'] = model_name
        
        # Collect relevant metrics
        model_summary = {
            'Model': model_name,
            'Mean AUC (Top 10)': top_10['auc_roc'].mean(),
            'Max AUC': top_10['auc_roc'].max(),
            'Min AUC (Top 10)': top_10['auc_roc'].min(),
            'Std AUC (Top 10)': top_10['auc_roc'].std(),
            'Mean Accuracy (Top 10)': top_10['accuracy'].mean(),
            'Mean F1 (Top 10)': top_10['f1_score'].mean() if 'f1_score' in top_10.columns else np.nan
        }
        
        all_model_data.append(model_summary)
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Create summary dataframe
summary_df = pd.DataFrame(all_model_data)
summary_df = summary_df.sort_values('Mean AUC (Top 10)', ascending=False)

# Save summary table
summary_df.to_csv(report_path / 'model_performance_summary.csv', index=False)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# 1. Bar plot of Mean AUC for all models
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(summary_df['Model'], summary_df['Mean AUC (Top 10)'])
ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
ax1.set_ylabel('Mean AUC (Top 10 Iterations)')
ax1.set_title('Model Performance Comparison - Mean AUC', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.8)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

# 2. Box plot showing AUC distribution for each model
ax2 = plt.subplot(2, 3, 2)
# Collect AUC data for box plot
auc_data_for_boxplot = []
model_names_for_boxplot = []

for model_name, config in models_config.items():
    file_path = result_num_path / config['path'] / config['file']
    try:
        df = pd.read_csv(file_path)
        top_10 = df.nlargest(10, 'auc_roc')
        auc_data_for_boxplot.append(top_10['auc_roc'].values)
        model_names_for_boxplot.append(model_name)
    except:
        pass

ax2.boxplot(auc_data_for_boxplot, labels=model_names_for_boxplot)
ax2.set_xticklabels(model_names_for_boxplot, rotation=45, ha='right')
ax2.set_ylabel('AUC ROC')
ax2.set_title('AUC Distribution (Top 10 Iterations)', fontsize=14, fontweight='bold')

# 3. Scatter plot: AUC vs Accuracy
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(summary_df['Mean Accuracy (Top 10)'], 
                      summary_df['Mean AUC (Top 10)'],
                      s=200, alpha=0.6, edgecolors='black')

# Add model labels
for idx, row in summary_df.iterrows():
    ax3.annotate(row['Model'], 
                 (row['Mean Accuracy (Top 10)'], row['Mean AUC (Top 10)']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

ax3.set_xlabel('Mean Accuracy (Top 10)')
ax3.set_ylabel('Mean AUC (Top 10)')
ax3.set_title('AUC vs Accuracy Trade-off', fontsize=14, fontweight='bold')

# 4. Error bar plot showing variability
ax4 = plt.subplot(2, 3, 4)
x_pos = np.arange(len(summary_df))
ax4.errorbar(x_pos, summary_df['Mean AUC (Top 10)'], 
             yerr=summary_df['Std AUC (Top 10)'],
             fmt='o', capsize=5, capthick=2)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
ax4.set_ylabel('Mean AUC ± Std')
ax4.set_title('Model Performance Variability', fontsize=14, fontweight='bold')

# 5. Radar chart for top 5 models
ax5 = plt.subplot(2, 3, 5, projection='polar')
top_5_models = summary_df.head(5)

# Prepare data for radar chart
categories = ['AUC', 'Accuracy', 'F1 Score']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

for idx, row in top_5_models.iterrows():
    values = [
        row['Mean AUC (Top 10)'],
        row['Mean Accuracy (Top 10)'],
        row['Mean F1 (Top 10)'] if not pd.isna(row['Mean F1 (Top 10)']) else 0
    ]
    values = np.concatenate([values, [values[0]]])
    ax5.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
    ax5.fill(angles, values, alpha=0.25)

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories)
ax5.set_ylim(0, 1)
ax5.set_title('Top 5 Models Performance Profile', fontsize=14, fontweight='bold', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 6. Heatmap of model performance metrics
ax6 = plt.subplot(2, 3, 6)
heatmap_data = summary_df[['Mean AUC (Top 10)', 'Mean Accuracy (Top 10)', 'Mean F1 (Top 10)']].T
heatmap_data.columns = summary_df['Model']

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
            cbar_kws={'label': 'Score'}, ax=ax6)
ax6.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(report_path / 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second figure for detailed performance analysis
fig2 = plt.figure(figsize=(16, 10))

# 1. Performance ranking
ax1 = plt.subplot(2, 2, 1)
y_pos = np.arange(len(summary_df))
ax1.barh(y_pos, summary_df['Mean AUC (Top 10)'])
ax1.set_yticks(y_pos)
ax1.set_yticklabels(summary_df['Model'])
ax1.set_xlabel('Mean AUC (Top 10 Iterations)')
ax1.set_title('Model Ranking by AUC Performance', fontsize=14, fontweight='bold')

# Add value labels
for i, v in enumerate(summary_df['Mean AUC (Top 10)']):
    ax1.text(v + 0.005, i, f'{v:.3f}', va='center')

# 2. Model performance by metric type
ax2 = plt.subplot(2, 2, 2)
metrics = ['Mean AUC (Top 10)', 'Max AUC', 'Mean Accuracy (Top 10)']
x = np.arange(len(summary_df))
width = 0.25

for i, metric in enumerate(metrics):
    offset = (i - 1) * width
    ax2.bar(x + offset, summary_df[metric], width, label=metric)

ax2.set_xlabel('Models')
ax2.set_ylabel('Score')
ax2.set_title('Multiple Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
ax2.legend()

# 3. Stability analysis (coefficient of variation)
ax3 = plt.subplot(2, 2, 3)
summary_df['CV'] = summary_df['Std AUC (Top 10)'] / summary_df['Mean AUC (Top 10)']
ax3.bar(summary_df['Model'], summary_df['CV'])
ax3.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
ax3.set_ylabel('Coefficient of Variation')
ax3.set_title('Model Stability (Lower is Better)', fontsize=14, fontweight='bold')

# 4. Performance gap analysis
ax4 = plt.subplot(2, 2, 4)
summary_df['Performance Gap'] = summary_df['Max AUC'] - summary_df['Min AUC (Top 10)']
ax4.bar(summary_df['Model'], summary_df['Performance Gap'])
ax4.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
ax4.set_ylabel('AUC Range (Max - Min in Top 10)')
ax4.set_title('Performance Consistency', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(report_path / 'detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations generated successfully!")
print(f"Summary saved to: {report_path / 'model_performance_summary.csv'}")
print(f"Main visualization saved to: {report_path / 'comprehensive_model_comparison.png'}")
print(f"Detailed analysis saved to: {report_path / 'detailed_performance_analysis.png'}")