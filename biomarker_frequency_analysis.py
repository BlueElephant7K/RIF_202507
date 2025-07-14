import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# Dictionary to store all biomarkers from each model
model_biomarkers = {
    'random_forest': {},
    'xgboost': {},
    'lightgbm': {},
    'svm': {},
    'logistic_regression': {},
    'linear_regression': {},
    'naive_bayes': {},
    'knn': {},
    'lda_qda': {}
}

# Process each model's data
# Random Forest
rf_genes = [
    ('PLSCR1', 60), ('TOB1', 60), ('USP17L7', 60), ('MAP1LC3B', 60), ('HTATIP2', 60),
    ('TAGLN2', 60), ('CAPZA3', 60), ('LINC01565', 60), ('ABCA7', 60), ('FN3KRP', 60),
    ('FAM210A', 60), ('PCCA', 60), ('ALDH2', 60), ('TMEM210', 60), ('SLC17A4', 60),
    ('RTN3', 60), ('TMBIM1', 60), ('PLXNA4', 60), ('LRP2', 60), ('RIOK3', 60),
    ('HIBADH', 60), ('ARG2', 60), ('RETREG1', 60), ('CLDN24', 60), ('RMND5A', 60),
    ('MACIR', 60), ('LIMK2', 60), ('DBR1', 60), ('MCM4', 60), ('MOCOS', 60),
    ('LAPTM4B', 60), ('HSPB8', 60), ('NPAS3', 60), ('GABARAPL1', 60), ('ZNF165', 60),
    ('EIF1', 60), ('SDCBP2', 60), ('DUSP1', 60), ('NUP43', 60), ('RNU4-49P', 60),
    ('TTC39C', 60), ('CHP1', 60), ('JPT1', 60), ('SCCPDH', 60), ('LINC02104', 60),
    ('MAOA', 60), ('WLS', 60), ('SERPINB1', 60), ('CD55', 60), ('SNRNP25', 60),
    ('HROB', 60), ('ARL6IP1', 60), ('RAB33B', 60), ('ZNF284', 60), ('ARRDC3', 60),
    ('CAMK2D', 60), ('CTNNA1', 60), ('NPHP4', 60), ('LRRC52', 60), ('ATF3', 60),
    ('ADSS2', 60), ('LINC01358', 60), ('H3C11', 60), ('LINC02340', 60), ('ATL3', 60),
    ('PLS3', 57), ('EPHA2', 57), ('NCCRP1', 57), ('MCM5', 57), ('RNU6-1209P', 57)
]

# XGBoost
xgb_genes = [
    ('LINC01565', 60), ('TMEM210', 60), ('ARRDC3', 60), ('ZNF284', 60), ('GABARAPL1', 60),
    ('MOCOS', 60), ('SNRNP25', 60), ('ZNF165', 57), ('DUSP1', 57), ('CTNNA1', 57),
    ('LINC02340', 57), ('EIF1', 57), ('VWA5A', 57), ('LIMK2', 57), ('MAOA', 57),
    ('SDCBP2', 57), ('RNU4-49P', 57), ('ATF3', 57), ('ARG2', 57), ('TTC39C', 57),
    ('NPHP4', 57), ('SPRING1', 54), ('FN3KRP', 54), ('RMND5A', 54), ('SCCPDH', 54),
    ('ATL3', 54), ('ALDH2', 54), ('FAM210A', 54), ('JPT1', 54), ('RAB33B', 54),
    ('ARL6IP1', 54), ('CREM', 54), ('SEPTIN10', 54), ('RTN3', 54), ('ABCA7', 54),
    ('CAMK2D', 54), ('TOB1', 54), ('LRRC52', 54), ('MAP1LC3B', 54), ('ADSS2', 54),
    ('DBR1', 54), ('TMBIM1', 54), ('LGALS7B', 54), ('CHP1', 54), ('HTATIP2', 54),
    ('MCM4', 54), ('HROB', 54), ('WLS', 54), ('RIOK3', 54), ('HSPB8', 51),
    ('NDRG1', 51), ('NAMPT', 51), ('TMEM237', 51), ('C6orf62', 51), ('CAMKV', 51)
]

# LightGBM - identical to XGBoost
lgbm_genes = xgb_genes.copy()

# SVM
svm_genes = [
    ('PLSCR1', 80), ('TOB1', 80), ('USP17L7', 80), ('MAP1LC3B', 80), ('HTATIP2', 80),
    ('TAGLN2', 80), ('CAPZA3', 80), ('LINC01565', 80), ('ABCA7', 80), ('FN3KRP', 80),
    ('FAM210A', 80), ('PCCA', 80), ('ALDH2', 80), ('TMEM210', 80), ('SLC17A4', 80),
    ('RTN3', 80), ('TMBIM1', 80), ('PLXNA4', 80), ('LRP2', 80), ('RIOK3', 80),
    ('HIBADH', 80), ('ARG2', 80), ('RETREG1', 80), ('CLDN24', 80), ('RMND5A', 80),
    ('MACIR', 80), ('LIMK2', 80), ('DBR1', 80), ('MCM4', 80), ('MOCOS', 80),
    ('LAPTM4B', 80), ('HSPB8', 80), ('NPAS3', 80), ('GABARAPL1', 80), ('ZNF165', 80),
    ('EIF1', 80), ('SDCBP2', 80), ('DUSP1', 80), ('NUP43', 80), ('RNU4-49P', 80),
    ('TTC39C', 80), ('CHP1', 80), ('JPT1', 80), ('SCCPDH', 80), ('LINC02104', 80),
    ('MAOA', 80), ('WLS', 80), ('SERPINB1', 80), ('CD55', 80), ('SNRNP25', 80),
    ('HROB', 80), ('ARL6IP1', 80), ('RAB33B', 80), ('ZNF284', 80), ('ARRDC3', 80),
    ('CAMK2D', 80), ('CTNNA1', 80), ('NPHP4', 80), ('LRRC52', 80), ('ATF3', 80),
    ('ADSS2', 80), ('LINC01358', 80), ('H3C11', 80), ('LINC02340', 80), ('ATL3', 80),
    ('PLS3', 76), ('EPHA2', 76), ('NCCRP1', 76), ('MCM5', 76), ('RNU6-1209P', 76)
]

# Logistic Regression
lr_genes = [
    ('PLSCR1', 60), ('TOB1', 60), ('USP17L7', 60), ('MAP1LC3B', 60), ('HTATIP2', 60),
    ('TAGLN2', 60), ('CAPZA3', 60), ('LINC01565', 60), ('ABCA7', 60), ('FN3KRP', 60),
    ('FAM210A', 60), ('PCCA', 60), ('ALDH2', 60), ('TMEM210', 60), ('SLC17A4', 60),
    ('RTN3', 60), ('TMBIM1', 60), ('PLXNA4', 60), ('LRP2', 60), ('RIOK3', 60),
    ('HIBADH', 60), ('ARG2', 60), ('RETREG1', 60), ('CLDN24', 60), ('RMND5A', 60),
    ('MACIR', 60), ('LIMK2', 60), ('DBR1', 60), ('MCM4', 60), ('MOCOS', 60),
    ('LAPTM4B', 60), ('HSPB8', 60), ('NPAS3', 60), ('GABARAPL1', 60), ('ZNF165', 60),
    ('EIF1', 60), ('SDCBP2', 60), ('DUSP1', 60), ('NUP43', 60), ('RNU4-49P', 60),
    ('TTC39C', 60), ('CHP1', 60), ('JPT1', 60), ('SCCPDH', 60), ('LINC02104', 60),
    ('MAOA', 60), ('WLS', 60), ('SERPINB1', 60), ('CD55', 60), ('SNRNP25', 60),
    ('HROB', 60), ('ARL6IP1', 60), ('RAB33B', 60), ('ZNF284', 60), ('ARRDC3', 60),
    ('CAMK2D', 60), ('CTNNA1', 60), ('NPHP4', 60), ('LRRC52', 60), ('ATF3', 60),
    ('ADSS2', 60), ('LINC01358', 60), ('H3C11', 60), ('LINC02340', 60), ('ATL3', 60),
    ('PLS3', 57), ('EPHA2', 57), ('NCCRP1', 57), ('MCM5', 57), ('RNU6-1209P', 57)
]

# Linear Regression
linear_genes = [
    ('PLSCR1', 120), ('TOB1', 120), ('USP17L7', 120), ('MAP1LC3B', 120), ('HTATIP2', 120),
    ('TAGLN2', 120), ('CAPZA3', 120), ('LINC01565', 120), ('ABCA7', 120), ('FN3KRP', 120),
    ('FAM210A', 120), ('PCCA', 120), ('ALDH2', 120), ('TMEM210', 120), ('SLC17A4', 120),
    ('RTN3', 120), ('TMBIM1', 120), ('PLXNA4', 120), ('LRP2', 120), ('RIOK3', 120),
    ('HIBADH', 120), ('ARG2', 120), ('RETREG1', 120), ('CLDN24', 120), ('RMND5A', 120),
    ('MACIR', 120), ('LIMK2', 120), ('DBR1', 120), ('MCM4', 120), ('MOCOS', 120),
    ('LAPTM4B', 120), ('HSPB8', 120), ('NPAS3', 120), ('GABARAPL1', 120), ('ZNF165', 120),
    ('EIF1', 120), ('SDCBP2', 120), ('DUSP1', 120), ('NUP43', 120), ('RNU4-49P', 120),
    ('TTC39C', 120), ('CHP1', 120), ('JPT1', 120), ('SCCPDH', 120), ('LINC02104', 120),
    ('MAOA', 120), ('WLS', 120), ('SERPINB1', 120), ('CD55', 120), ('SNRNP25', 120),
    ('HROB', 120), ('ARL6IP1', 120), ('RAB33B', 120), ('ZNF284', 120), ('ARRDC3', 120),
    ('CAMK2D', 120), ('CTNNA1', 120), ('NPHP4', 120), ('LRRC52', 120), ('ATF3', 120),
    ('ADSS2', 120), ('LINC01358', 120), ('H3C11', 120), ('LINC02340', 120), ('ATL3', 120),
    ('PLS3', 114), ('EPHA2', 114), ('NCCRP1', 114), ('MCM5', 114), ('RNU6-1209P', 114)
]

# Naive Bayes - same as Logistic Regression
nb_genes = lr_genes.copy()

# KNN
knn_genes = [
    ('PLSCR1', 100), ('TOB1', 100), ('USP17L7', 100), ('MAP1LC3B', 100), ('HTATIP2', 100),
    ('TAGLN2', 100), ('CAPZA3', 100), ('LINC01565', 100), ('ABCA7', 100), ('FN3KRP', 100),
    ('FAM210A', 100), ('PCCA', 100), ('ALDH2', 100), ('TMEM210', 100), ('SLC17A4', 100),
    ('RTN3', 100), ('TMBIM1', 100), ('PLXNA4', 100), ('LRP2', 100), ('RIOK3', 100),
    ('HIBADH', 100), ('ARG2', 100), ('RETREG1', 100), ('CLDN24', 100), ('RMND5A', 100),
    ('MACIR', 100), ('LIMK2', 100), ('DBR1', 100), ('MCM4', 100), ('MOCOS', 100),
    ('LAPTM4B', 100), ('HSPB8', 100), ('NPAS3', 100), ('GABARAPL1', 100), ('ZNF165', 100),
    ('EIF1', 100), ('SDCBP2', 100), ('DUSP1', 100), ('NUP43', 100), ('RNU4-49P', 100),
    ('TTC39C', 100), ('CHP1', 100), ('JPT1', 100), ('SCCPDH', 100), ('LINC02104', 100),
    ('MAOA', 100), ('WLS', 100), ('SERPINB1', 100), ('CD55', 100), ('SNRNP25', 100),
    ('HROB', 100), ('ARL6IP1', 100), ('RAB33B', 100), ('ZNF284', 100), ('ARRDC3', 100),
    ('CAMK2D', 100), ('CTNNA1', 100), ('NPHP4', 100), ('LRRC52', 100), ('ATF3', 100),
    ('ADSS2', 100), ('LINC01358', 100), ('H3C11', 100), ('LINC02340', 100), ('ATL3', 100),
    ('PLS3', 95), ('EPHA2', 95), ('NCCRP1', 95), ('MCM5', 95), ('RNU6-1209P', 95)
]

# LDA/QDA - same as SVM
lda_qda_genes = svm_genes.copy()

# Populate model dictionaries
for gene, freq in rf_genes[:66]:  # Top 66 genes with 100% frequency
    model_biomarkers['random_forest'][gene] = freq

for gene, freq in xgb_genes[:60]:
    model_biomarkers['xgboost'][gene] = freq
    model_biomarkers['lightgbm'][gene] = freq

for gene, freq in svm_genes[:66]:
    model_biomarkers['svm'][gene] = freq
    model_biomarkers['lda_qda'][gene] = freq

for gene, freq in lr_genes[:66]:
    model_biomarkers['logistic_regression'][gene] = freq
    model_biomarkers['naive_bayes'][gene] = freq

for gene, freq in linear_genes[:66]:
    model_biomarkers['linear_regression'][gene] = freq

for gene, freq in knn_genes[:66]:
    model_biomarkers['knn'][gene] = freq

# Count gene appearances across models
gene_appearances = defaultdict(int)
gene_model_count = defaultdict(list)

for model, genes in model_biomarkers.items():
    for gene in genes:
        gene_appearances[gene] += 1
        gene_model_count[gene].append(model)

# Sort genes by frequency across models
sorted_genes = sorted(gene_appearances.items(), key=lambda x: x[1], reverse=True)

# Create comprehensive report
print("="*80)
print("COMPREHENSIVE BIOMARKER FREQUENCY ANALYSIS")
print("="*80)
print()

print("1. TOP 10 MOST FREQUENTLY SELECTED GENES ACROSS ALL MODELS")
print("-"*60)
print(f"{'Rank':<6}{'Gene':<20}{'Models':<10}{'Model Names'}")
print("-"*60)
for i, (gene, count) in enumerate(sorted_genes[:10], 1):
    models = ", ".join(gene_model_count[gene])
    print(f"{i:<6}{gene:<20}{count:<10}{models}")

print("\n2. GENES APPEARING IN ALL 9 MODELS")
print("-"*60)
universal_genes = [gene for gene, count in sorted_genes if count == 9]
print(f"Total: {len(universal_genes)} genes")
for i, gene in enumerate(universal_genes, 1):
    if i % 5 == 0:
        print(gene)
    else:
        print(f"{gene:<15}", end="")
if len(universal_genes) % 5 != 0:
    print()

print("\n3. TOP 5 GENES PER MODEL TYPE")
print("-"*60)
for model in model_biomarkers:
    print(f"\n{model.upper().replace('_', ' ')}:")
    top_genes = list(model_biomarkers[model].keys())[:5]
    for i, gene in enumerate(top_genes, 1):
        print(f"  {i}. {gene}")

print("\n4. FREQUENCY DISTRIBUTION")
print("-"*60)
freq_dist = Counter(gene_appearances.values())
for models_count in sorted(freq_dist.keys(), reverse=True):
    gene_count = freq_dist[models_count]
    print(f"Genes appearing in {models_count} models: {gene_count} genes")

print("\n5. UNIQUE GENES BY MODEL TYPE")
print("-"*60)
for model in model_biomarkers:
    unique_genes = []
    for gene in model_biomarkers[model]:
        if gene_appearances[gene] == 1:
            unique_genes.append(gene)
    if unique_genes:
        print(f"\n{model.upper().replace('_', ' ')}: {len(unique_genes)} unique genes")
        for gene in unique_genes[:3]:
            print(f"  - {gene}")
        if len(unique_genes) > 3:
            print(f"  ... and {len(unique_genes) - 3} more")

# Save results to file
with open('/Users/nan/Documents/blueelephant/RIF_202507/biomarker_analysis_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE BIOMARKER FREQUENCY ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. TOP 10 MOST FREQUENTLY SELECTED GENES ACROSS ALL MODELS\n")
    f.write("-"*60 + "\n")
    f.write(f"{'Rank':<6}{'Gene':<20}{'Models':<10}{'Model Names'}\n")
    f.write("-"*60 + "\n")
    for i, (gene, count) in enumerate(sorted_genes[:10], 1):
        models = ", ".join(gene_model_count[gene])
        f.write(f"{i:<6}{gene:<20}{count:<10}{models}\n")
    
    f.write("\n2. GENES APPEARING IN ALL 9 MODELS\n")
    f.write("-"*60 + "\n")
    f.write(f"Total: {len(universal_genes)} genes\n\n")
    for i, gene in enumerate(universal_genes, 1):
        f.write(f"{gene:<15}")
        if i % 5 == 0:
            f.write("\n")
    if len(universal_genes) % 5 != 0:
        f.write("\n")
    
    f.write("\n3. DETAILED GENE FREQUENCY TABLE\n")
    f.write("-"*60 + "\n")
    f.write(f"{'Gene':<20}{'Frequency':<12}{'Models'}\n")
    f.write("-"*60 + "\n")
    for gene, count in sorted_genes[:50]:  # Top 50 genes
        models = ", ".join(gene_model_count[gene])
        f.write(f"{gene:<20}{count:<12}{models}\n")

print("\nAnalysis complete! Results saved to biomarker_analysis_results.txt")