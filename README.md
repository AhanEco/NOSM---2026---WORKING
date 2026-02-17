# Network-Oligopoly Survival Model (NOSM) 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![NetworkX](https://img.shields.io/badge/Graph-NetworkX-green)

An empirical system that predicts startup survival by analyzing the **Global Investor Network**. 
Instead of looking at a startup in isolation, NOSM models it as a node in a complex graph of capital and influence, utilizing **Social Network Analysis (SNA)** and **Industrial Organization Economics**.

---

## 📊 Key Findings
> **"It's not just what you know, but who you know—and how diverse they are."**

Our analysis of over **32,000 companies** and **95,000 investment events** reveals:
1. **Weak Ties Matter**: Startups with diverse, disconnected investors (High Weak Tie Ratio) are more likely to survive than those in tight cliques.
2. **Oligopolies Require Brokerage**: In concentrated industries (High HHI), acting as a "Broker" between clusters is the strongest predictor of breakout success.
3. **Cash isn't everything**: Network structural features often outweigh raw capital availability in predicting survival.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NOSM.git
   cd NOSM
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy networkx xgboost scikit-learn streamlit scipy matplotlib
   ```

3. **Get the Data**:
   The project requires `companies.csv` and `investments.csv` provided by [Crunchbase Data](https://github.com/notpeter/crunchbase-data). Place them in the `data/` directory.
   *(Note: The repository includes a `download_data` script or manual download instructions)*.

---

## 🚀 Usage

### 1. Train the Model
Run the data pipeline to build the graph, calculate network metrics (HHI, Clustering, Centrality), and train the XGBoost classifier.
```bash
python train_model.py
```
*Outputs: `data/processed_data.csv`, `models/model.json`*

### 2. Run the Web Interface
Launch the interactive dashboard to simulate startup scenarios.
```bash
streamlit run app.py
```
*Access at: `http://localhost:8501`*

---

## 🧠 Methodology

### 1. Network Construction (Bipartite Projection)
We model the ecosystem as a **Bipartite Graph** (Investors $\leftrightarrow$ Startups). We project this into a **Startup-Startup Graph** where edges represent shared investors.
- **Nodes**: 32,069 Startups
- **Edges**: ~2.9 Million (Shared Investor connections)

### 2. Feature Engineering
We calculate advanced graph-theoretic metrics:
- **Weak Tie Ratio**: Inverse of structural redundancy (measure of information novelty).
- **Clustering Coefficient**: Measure of "Safety" and trust (Clique-iness).
- **Industry HHI**: Herfindahl-Hirschman Index for market concentration.
- **Breakout Velocity**: Interaction term combining Network Diversity and Capital.

### 3. Machine Learning (XGBoost)
We use a Gradient Boosted Tree with `scale_pos_weight` to handle the class imbalance (Failure is rare in the dataset relative to "Operating").
- **Recall (Failure Identification)**: **47%** (vs Baseline 0%)

---

## 📂 Project Structure

```
NOSM/
├── app.py                # Streamlit Web Application
├── train_model.py        # ML Pipeline (Graph Construction + Training)
├── data/                 # Raw and Processed Data
│   ├── companies.csv
│   └── investments.csv
├── models/               # Saved XGBoost Models
│   └── model.json
├── research_paper.tex    # Full Academic Paper (LaTeX)
└── README.md             # This file
```

---

## 🎮 Case Study: The Gaming Oligopoly
The repository includes a detailed analysis of the **Sony vs. Microsoft** war.
- **Sony**: High Clustering (Strong Ties with Naughty Dog, Insomniac). Strategy: **Quality/Prestige**.
- **Microsoft**: High Weak Ties (Bridging PC, Cloud, Console via Activision/Bethesda). Strategy: **Brokerage/Scale**.

See `research_paper.pdf` (compile from `.tex`) for the full case study.

---

## License
MIT License
