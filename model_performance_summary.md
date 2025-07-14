# Model Performance Analysis Summary

## Overview
This analysis covers 9 different model types with 20 iterations each. Performance is primarily evaluated using AUC (Area Under Curve) scores, which is a robust metric for classification performance.

## Model Rankings by Performance

### 1. **Random Forest (Best Overall)**
- **AUC Range**: 0.4130 - 0.7376
- **Top 10 Iterations AUC Scores**: 
  1. Iteration 11, ExtraTrees: 0.7065
  2. Iteration 11, RF_Deeper: 0.6804
  3. Iteration 11, RandomForest: 0.6674
  4. Iteration 17, RF_Deeper: 0.6975
  5. Iteration 17, ExtraTrees: 0.6833
  6. Iteration 17, RandomForest: 0.6754
  7. Iteration 7, RF_Deeper: 0.6750
  8. Iteration 7, RandomForest: 0.6540
  9. Iteration 4, ExtraTrees: 0.6420
  10. Iteration 18, ExtraTrees: 0.6536
- **Mean AUC of Top 10**: 0.6731

### 2. **LightGBM**
- **AUC Range**: 0.3551 - 0.7478
- **Top 10 Iterations AUC Scores**:
  1. Iteration 0, LightGBM_Accurate: 0.7478
  2. Iteration 17, LightGBM_Default: 0.7377
  3. Iteration 17, LightGBM_Accurate: 0.6964
  4. Iteration 17, LightGBM_Fast: 0.6587
  5. Iteration 13, LightGBM_Fast: 0.6699
  6. Iteration 13, LightGBM_Default: 0.6591
  7. Iteration 13, LightGBM_Accurate: 0.6688
  8. Iteration 18, LightGBM_Fast: 0.6428
  9. Iteration 6, LightGBM_Accurate: 0.6391
  10. Iteration 9, LightGBM_Default: 0.6637
- **Mean AUC of Top 10**: 0.6784

### 3. **XGBoost**
- **AUC Range**: 0.4178 - 0.7134
- **Top 10 Iterations AUC Scores**:
  1. Iteration 0, XGB_Regularized: 0.7134
  2. Iteration 0, XGB_Default: 0.6804
  3. Iteration 17, XGB_Default: 0.6583
  4. Iteration 17, XGB_Regularized: 0.6431
  5. Iteration 5, XGB_Default: 0.6482
  6. Iteration 6, XGB_Regularized: 0.6312
  7. Iteration 12, XGB_Deep: 0.6261
  8. Iteration 10, XGB_Deep: 0.6246
  9. Iteration 6, XGB_Default: 0.6214
  10. Iteration 4, XGB_Deep: 0.6144
- **Mean AUC of Top 10**: 0.6461

### 4. **SVM**
- **AUC Range**: 0.4246 - 0.7007
- **Top 10 Iterations AUC Scores**:
  1. Iteration 11, SVM_Poly: 0.7007
  2. Iteration 0, SVM_RBF: 0.6696
  3. Iteration 17, SVM_RBF: 0.6630
  4. Iteration 4, SVM_Nu_SVM: 0.6551
  5. Iteration 17, SVM_Poly: 0.6486
  6. Iteration 4, SVM_Poly: 0.6457
  7. Iteration 4, SVM_RBF: 0.6413
  8. Iteration 15, SVM_Poly: 0.6391
  9. Iteration 18, SVM_Poly: 0.6043
  10. Iteration 17, SVM_Nu_SVM: 0.6333
- **Mean AUC of Top 10**: 0.6501

### 5. **KNN**
- **AUC Range**: 0.4173 - 0.6637
- **Top 10 Iterations AUC Scores**:
  1. Iteration 9, KNN_cosine: 0.6638
  2. Iteration 11, KNN_adaptive: 0.6464
  3. Iteration 6, KNN_adaptive: 0.6391
  4. Iteration 3, SVM_manhattan: 0.6094
  5. Iteration 11, KNN_cosine: 0.6083
  6. Iteration 7, KNN_cosine: 0.6080
  7. Iteration 0, KNN_adaptive: 0.6043
  8. Iteration 10, KNN_adaptive: 0.6028
  9. Iteration 8, KNN_k5_euclidean: 0.5986
  10. Iteration 13, KNN_adaptive: 0.6076
- **Mean AUC of Top 10**: 0.6188

### 6. **Logistic Regression**
- **AUC Range**: 0.4324 - 0.6508
- **Top 10 Iterations AUC Scores**:
  1. Iteration 16, LR_L1: 0.6508
  2. Iteration 11, LR_ElasticNet: 0.6290
  3. Iteration 0, LR_L1: 0.6275
  4. Iteration 11, LR_L1: 0.6152
  5. Iteration 4, LR_L1: 0.6145
  6. Iteration 4, LR_L2: 0.6080
  7. Iteration 2, LR_L1: 0.6043
  8. Iteration 0, LR_L2: 0.6029
  9. Iteration 15, LR_Regularized: 0.6025
  10. Iteration 5, LR_Regularized: 0.5967
- **Mean AUC of Top 10**: 0.6151

### 7. **Naive Bayes**
- **AUC Range**: 0.3595 - 0.6428
- **Top 10 Iterations AUC Scores**:
  1. Iteration 18, Multinomial NB: 0.6428
  2. Iteration 17, Multinomial NB: 0.6297
  3. Iteration 16, Bernoulli NB: 0.6268
  4. Iteration 18, Bernoulli NB: 0.6210
  5. Iteration 11, Multinomial NB: 0.6094
  6. Iteration 15, Multinomial NB: 0.6072
  7. Iteration 1, Multinomial NB: 0.6014
  8. Iteration 18, Gaussian NB: 0.5960
  9. Iteration 16, Multinomial NB: 0.5928
  10. Iteration 19, Multinomial NB: 0.5917
- **Mean AUC of Top 10**: 0.6119

### 8. **Linear Regression**
- **AUC Range**: 0.3551 - 0.7377
- **Top 10 Iterations AUC Scores**:
  1. Iteration 17, ElasticNet: 0.7377 
  2. Iteration 4, ElasticNet: 0.6304
  3. Iteration 17, Linear_Regression: 0.6210
  4. Iteration 3, ElasticNet: 0.6131
  5. Iteration 11, ElasticNet: 0.6116
  6. Iteration 11, Ridge: 0.6101
  7. Iteration 11, Linear_Regression: 0.6094
  8. Iteration 0, Linear_Regression: 0.6072
  9. Iteration 0, Ridge: 0.6080
  10. Iteration 2, ElasticNet: 0.5913
- **Mean AUC of Top 10**: 0.6240

### 9. **LDA/QDA**
- **AUC Range**: 0.3456 - 0.6261
- **Top 10 Iterations AUC Scores**:
  1. Iteration 11, LDA: 0.6261
  2. Iteration 0, LDA: 0.6203
  3. Iteration 16, LDA: 0.6054
  4. Iteration 13, QDA_Regularized: 0.5971
  5. Iteration 13, QDA: 0.5899
  6. Iteration 18, QDA: 0.5891
  7. Iteration 4, LDA: 0.5877
  8. Iteration 18, LDA: 0.5804
  9. Iteration 7, QDA: 0.5787
  10. Iteration 17, LDA: 0.5739
- **Mean AUC of Top 10**: 0.5949

## Key Insights

1. **Best Performing Models**: LightGBM and Random Forest show the highest potential with maximum AUC scores above 0.70, indicating strong discriminative ability.

2. **Most Consistent**: Random Forest models (especially ExtraTrees variants) show good consistency with multiple iterations in the top 10.

3. **High Variance Models**: Linear Regression shows surprising variance with one iteration achieving 0.7377 AUC (likely an outlier or specific favorable data split).

4. **Stable Mid-Performers**: SVM and XGBoost show stable mid-range performance with good potential for optimization.

5. **Lower Performers**: LDA/QDA and Naive Bayes consistently show lower AUC scores, suggesting they may not be well-suited for this particular dataset.

## Recommendations

1. **Focus on LightGBM and Random Forest**: These models show the best overall performance and should be prioritized for hyperparameter tuning.

2. **Investigate High-Performing Iterations**: The specific configurations that led to high AUC scores (e.g., LightGBM_Accurate, ExtraTrees) should be analyzed further.

3. **Ensemble Methods**: Consider creating an ensemble of the top-performing models (Random Forest, LightGBM, XGBoost) for potentially better results.

4. **Feature Engineering**: The variance in results suggests that feature selection or engineering could significantly impact model performance.