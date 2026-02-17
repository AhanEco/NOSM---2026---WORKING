import pandas as pd
import networkx as nx
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os
import scipy.sparse as sp

# Configuration
DATA_DIR = "data"
COMPANIES_FILE = os.path.join(DATA_DIR, "companies.csv")
INVESTMENTS_FILE = os.path.join(DATA_DIR, "investments.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_data.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "model.json")

# Relevant Industries
RELEVANT_INDUSTRIES = ['Software', 'Web', 'Mobile', 'Enterprise Software', 'E-Commerce', 'Games', 'Biotechnology']

def load_data():
    """Loads companies and investments data."""
    print("Loading data...")
    if not os.path.exists(COMPANIES_FILE) or not os.path.exists(INVESTMENTS_FILE):
        raise FileNotFoundError(f"Data files not found in {DATA_DIR}.")
        
    companies = pd.read_csv(COMPANIES_FILE, encoding='utf-8', on_bad_lines='skip')
    investments = pd.read_csv(INVESTMENTS_FILE, encoding='utf-8', on_bad_lines='skip')
    return companies, investments

def preprocess_companies(companies):
    """Filters companies and defines the target variable."""
    print("Preprocessing companies...")
    
    # Filter by industry
    companies['category_list'] = companies['category_list'].fillna('')
    mask = companies['category_list'].apply(lambda x: any(industry in x for industry in RELEVANT_INDUSTRIES))
    df = companies[mask].copy()
    
    # Define Target: Survival
    df['status'] = df['status'].fillna('operating') 
    df['target'] = df['status'].apply(lambda x: 0 if x == 'closed' else 1)
    
    print(f"Filtered to {len(df)} companies in relevant industries.")
    print(f"Target Distribution:\n{df['target'].value_counts()}")
    
    return df

def build_network_features_sparse(companies_df, investments_df):
    """Builds the graph using sparse matrices for efficiency."""
    print("Building network features (Sparse Matrix Optimization)...")
    
    # Filter investments to relevant companies
    relevant_permalinks = set(companies_df['permalink'])
    inv_df = investments_df[investments_df['company_permalink'].isin(relevant_permalinks)].copy()
    
    # Create mappings
    # Companies
    company_nodes = inv_df['company_permalink'].unique()
    company_to_idx = {node: i for i, node in enumerate(company_nodes)}
    
    # Investors
    investor_nodes = inv_df['investor_permalink'].unique()
    investor_to_idx = {node: i for i, node in enumerate(investor_nodes)}
    
    print(f"Graph Order: {len(company_nodes)} Companies, {len(investor_nodes)} Investors")
    print(f"Investments (Edges): {len(inv_df)}")
    
    # Build Bipartite Adjacency Matrix B (Rows=Companies, Cols=Investors)
    rows = inv_df['company_permalink'].map(company_to_idx).values
    cols = inv_df['investor_permalink'].map(investor_to_idx).values
    data = np.ones(len(inv_df))
    
    B = sp.coo_matrix((data, (rows, cols)), shape=(len(company_nodes), len(investor_nodes)))
    B_csr = B.tocsr()
    
    # Project to Company-Company Graph P = B * B.T
    # P[i, j] = number of common investors
    print("Projecting graph (Matrix Multiplication)...")
    P = B_csr.dot(B_csr.T)
    
    # Calculate Features from P
    print("Calculating Matrix Metrics...")
    
    # 1. Number of Investors (Diagonal of P)
    num_investors = P.diagonal()
    
    # 2. Degree Centrality (Weighted Degree in Projected Graph)
    # Sum of row - diagonal (self-loop)
    # This represents total shared connections (if company A shares 2 investors with B, it adds 2 to degree).
    # Unweighted degree would be count of non-zero elements.
    weighted_degree = np.array(P.sum(axis=1)).flatten() - num_investors
    
    # 3. Clustering & Weak Ties
    # If P is too dense, converting to NX is slow.
    num_edges = P.nnz
    print(f"Projected Graph Edges (approx): {num_edges}")
    
    clustering_coeffs = np.zeros(len(company_nodes))
    weak_tie_ratios = np.zeros(len(company_nodes))
    
    # Calculate Redundancy Proxy for all nodes first
    # Redundancy = Weighted Degree / Num Investors
    # High Redundancy = Many shared investors per investor = Clique-ish = Strong Ties
    # Low Redundancy = Few shared investors per investor = Bridge-ish = Weak Ties
    # Avoid division by zero
    redundancy = weighted_degree / (num_investors + 1)
    weak_tie_ratios = 1.0 / (redundancy + 0.1) # Inverse of redundancy
    
    if num_edges < 1000000: # Threshold for NX conversion
        print("Graph size manageable. Converting to NetworkX for structural metrics...")
        # Remove self-loops for NX
        P.setdiag(0)
        P.eliminate_zeros()
        
        G = nx.from_scipy_sparse_array(P)
        G = nx.relabel_nodes(G, {i: node for i, node in enumerate(company_nodes)})
        
        # Clustering
        print("Calculating Clustering...")
        clustering = nx.clustering(G, weight='weight')
        
        for i, node in enumerate(company_nodes):
            clustering_coeffs[i] = clustering.get(node, 0)
            
    else:
        print("Graph too large for NetworkX. Using approximation metrics.")
        # Clustering Proxy: correlated with Redundancy?
        # Let's use Redundancy as a proxy for Clustering too (High Redundancy ~ High Clustering)
        clustering_coeffs = redundancy / (redundancy.max() + 0.01) # Normalize

    # Create Metrics DataFrame
    metrics = pd.DataFrame({
        'permalink': company_nodes,
        'clustering_coefficient': clustering_coeffs,
        'degree_centrality': weighted_degree, # Using weighted degree as centrality
        'weak_tie_ratio': weak_tie_ratios,
        'num_investors': num_investors
    })
    
    return metrics

def calculate_hhi(df):
    """Calculates Industry HHI."""
    print("Calculating Industry HHI...")
    
    df['primary_category'] = df['category_list'].astype(str).apply(lambda x: x.split('|')[1] if '|' in x and len(x.split('|')) > 1 else 'Unknown')
    
    def clean_funding(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace('-', '').strip()
            return float(x) if x else 0.0
        return float(x) if pd.notnull(x) else 0.0
        
    df['funding_total_usd_clean'] = df['funding_total_usd'].apply(clean_funding)
    
    category_totals = df.groupby('primary_category')['funding_total_usd_clean'].transform('sum')
    df['market_share'] = df['funding_total_usd_clean'] / category_totals
    df['market_share'] = df['market_share'].fillna(0)
    
    hhi_df = df.groupby('primary_category')['market_share'].apply(lambda x: (x**2).sum()).reset_index(name='category_hhi')
    
    df = pd.merge(df, hhi_df, on='primary_category', how='left')
    df.rename(columns={'category_hhi': 'industry_hhi'}, inplace=True)
    
    return df

def feature_engineering(df, network_metrics):
    """Combines features."""
    print("Merging features...")
    
    full_df = pd.merge(df, network_metrics, on='permalink', how='left')
    
    # Fill missing
    for col in ['clustering_coefficient', 'degree_centrality', 'weak_tie_ratio', 'num_investors']:
        full_df[col] = full_df[col].fillna(0)
    
    # Working Capital
    full_df['working_capital'] = np.log1p(full_df['funding_total_usd_clean'])
    
    # Breakout Velocity
    full_df['breakout_velocity'] = (full_df['weak_tie_ratio'] * full_df['working_capital']) / (full_df['industry_hhi'] + 0.01)
    
    return full_df

def train_xgboost(df):
    """Trains XGBoost."""
    print("Training XGBoost...")
    
    features = ['working_capital', 'industry_hhi', 'clustering_coefficient', 'degree_centrality', 'weak_tie_ratio', 'breakout_velocity', 'num_investors']
    X = df[features]
    y = df['target']
    
    # Calculate scale_pos_weight
    # failure (0) is minority. survival (1) is majority.
    # We want to balance sensitivity.
    # To treat 0 equal to 1: scale_pos_weight = count(0) / count(1)
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    print(f"Class Imbalance: {num_neg} closed vs {num_pos} operating. scale_pos_weight={scale_weight:.4f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_weight)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nReport:\n", classification_report(y_test, y_pred))
    
    imps = model.feature_importances_
    features_sorted = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for f, v in features_sorted:
        print(f"{f}: {v:.5f}")
        
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

def main():
    try:
        companies, investments = load_data()
        
        companies = preprocess_companies(companies)
        companies = calculate_hhi(companies)
        
        network_metrics = build_network_features_sparse(companies, investments)
        
        final_df = feature_engineering(companies, network_metrics)
        
        final_df.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")
        
        train_xgboost(final_df)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
