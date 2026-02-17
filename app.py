import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import networkx as nx
import matplotlib.pyplot as plt
import os
import json

# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "model.json")

# Set Page Config
st.set_page_config(page_title="Startup Survival Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DATA_FILE)

def main():
    st.title("Network-Oligopoly Survival Model (NOSM)")
    st.markdown("### Predict your startup's probability of survival based on Industry & Network Strategy.")

    # Load Resources
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return

    # Sidebar inputs
    st.sidebar.header("Simulation Parameters")
    
    # 1. Industry Selection
    industries = df['primary_category'].unique()
    industry = st.sidebar.selectbox("Select Industry", options=industries)
    
    # Get Industry Stats
    industry_stats = df[df['primary_category'] == industry].iloc[0]
    hhi = industry_stats['industry_hhi']
    st.sidebar.markdown(f"**Industry HHI**: {hhi:.4f} (Concentration Index)")
    
    # 2. Capital
    funding = st.sidebar.slider("Funding Amount ($)", min_value=1000, max_value=100_000_000, value=1_000_000, step=1000)
    working_capital = np.log1p(funding)
    
    # 3. Network Strategy
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Network Strategy**")
    
    # Network Size (Degree)
    degree_percentile = st.sidebar.slider("Network Size (Percentile)", 0, 100, 50, help="How connected are you compared to peers?")
    degree = np.percentile(df['degree_centrality'], degree_percentile)
    
    # Structural Position (Weak Ties vs Clustering)
    # Slider from "Tight Clique" to "Brokerage/Open"
    # Tight Cipher = High Clustering, Low Weak Ties
    # Brokerage = Low Clustering, High Weak Ties
    strategy = st.sidebar.slider("Network Position", 0, 100, 50, format="%d%% Brokerage", help="0% = Tight Clique (Safety), 100% = Structural Holes (Innovation)")
    
    # Map strategy to metrics (Linear interpolation between min and max of dataset)
    min_wt, max_wt = df['weak_tie_ratio'].min(), df['weak_tie_ratio'].max()
    min_cl, max_cl = df['clustering_coefficient'].min(), df['clustering_coefficient'].max()
    
    # High Strategy = High Weak Ties, Low Clustering
    weak_tie_ratio = min_wt + (max_wt - min_wt) * (strategy / 100)
    clustering = max_cl - (max_cl - min_cl) * (strategy / 100)
    
    # Number of Investors (Correlated with size usually, but let's separate or infer)
    # Let's infer based on degree roughly or just take average for simplicty if not a direct input
    # Model uses `num_investors`. Let's assume proportional to degree with some factor.
    avg_investor_degree = df['degree_centrality'].mean() / df['num_investors'].mean() if df['num_investors'].mean() > 0 else 1
    num_investors = degree / avg_investor_degree
    
    # Interaction Term
    breakout_velocity = (weak_tie_ratio * working_capital) / (hhi + 0.01)
    
    # Prepare Input
    # Features: ['working_capital', 'industry_hhi', 'clustering_coefficient', 'degree_centrality', 'weak_tie_ratio', 'breakout_velocity', 'num_investors']
    input_data = pd.DataFrame([{
        'working_capital': working_capital,
        'industry_hhi': hhi,
        'clustering_coefficient': clustering,
        'degree_centrality': degree,
        'weak_tie_ratio': weak_tie_ratio,
        'breakout_velocity': breakout_velocity,
        'num_investors': num_investors
    }])
    
    # Prediction
    prob = model.predict_proba(input_data)[0][1]
    
    # Visualization Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Survival Result")
        st.metric(label="Survival Probability", value=f"{prob:.1%}", delta=f"{prob-0.5:.1%}")
        
        if prob > 0.7:
            st.success("High chance of survival! You have a strong breakout velocity.")
        elif prob > 0.4:
            st.warning("Moderate risk. Consider raising more capital or diversifying your network.")
        else:
            st.error("High risk of failure. Your network might be too insular or capital too low for this industry.")
            
        st.write("### Model Inputs")
        st.json(input_data.T.to_dict()[0])

    with col2:
        st.subheader("Network Visualization (Conceptual)")
        
        # Draw a small ego graph representing the strategy
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create a synthetic graph based on inputs
        # High Clustering = connected neighbors
        # High Weak Ties = disconnected neighbors
        G = nx.Graph()
        center = 0
        G.add_node(center)
        
        # Number of neighbors ~ Scaled Degree (capped forviz)
        n_neighbors = int(max(3, min(degree / 10, 20))) 
        neighbors = range(1, n_neighbors + 1)
        G.add_nodes_from(neighbors)
        for n in neighbors:
            G.add_edge(center, n)
            
        # Add edges between neighbors based on clustering input
        # Probability of edge between neighbors ~ Clustering Coefficient
        for i in neighbors:
            for j in neighbors:
                if i < j:
                    if np.random.random() < clustering:
                        G.add_edge(i, j)
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=False, node_color='skyblue', edge_color='gray', node_size=300)
        nx.draw_networkx_nodes(G, pos, nodelist=[center], node_color='orange', node_size=500)
        ax.set_title("Ego Network Structure")
        st.pyplot(fig)
        
        st.info(f"Visualizing a network with {int(clustering*100)}% clustering among {n_neighbors} connections.")

if __name__ == "__main__":
    main()
