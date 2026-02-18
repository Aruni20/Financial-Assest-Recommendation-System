# FAR-Trans: Hybrid AI for Personalized Financial Asset Recommendation
### A Multi-Modal Approach Combining Collaborative Filtering, Content Analysis, and Demographic Profiling for Retail Investors

## 🚀 Live Demo
**[Click Here to Launch App](https://financial-asset-recommendation-system.streamlit.app/)**
*(Note: If the link is not yet active, please deploy the app on Streamlit Community Cloud and update this URL)*

## Overview
**Can we engineer a financial advisor that balances historical trading patterns, asset fundamentals, and individual risk tolerance?**

This project answers that question by building a robust **Hybrid Recommendation Engine** using the **FAR-Trans dataset**. unlike traditional recommenders that rely solely on "people who bought X also bought Y," this system integrates **three distinct intelligent signals**: latent behavioral patterns (Collaborative Filtering), fundamental asset attributes (Content-Based), and investor compatibility scores (Demographic Profiling).

The result is a holistic, interpretable, and high-performance system that doesn't just predict what a user *might* buy, but suggests what they *should* buy based on their financial "DNA."

## Motivation
The modern retail investor faces a paradox of choice: thousands of assets, complex financial data, and limited personal capacity for analysis. Traditional brokerage apps often push "trending" stocks, leading to herding behavior and portfolio concentration risk.

**Behavioral Finance Perspective**:
Retail investors often exhibit *bounded rationality*—making suboptimal decisions due to cognitive limits. A purely data-driven recommender can act as a rational check, surfacing assets that align with a user's long-term goals and risk capacity rather than short-term market noise. This project aims to democratize access to sophisticated, personalized financial advice typically reserved for high-net-worth clients.

## Research Questions
1.  **Latent Structure**: Can Matrix Factorization (SVD) effectively de-noise sparse financial transaction matrices to find hidden trading patterns?
2.  **Fundamental Alignment**: Does adding content-based features (Sector, Industry, Profitability) solve the "Cold Start" problem for new assets?
3.  **Risk Calibration**: Can demographic heuristics (Risk Tolerance, Capital) act as a reliable filter to prevent unsuitable recommendations (e.g., preventing a conservative retiree from buying volatile crypto)?
4.  **Hybrid Synergy**: Does a weighted ensemble of these three components outperform individual baselines in ranking metrics like nDCG and Precision?

## Data Source: FAR-Trans
The system is built on the **FAR-Trans (Financial Asset Recommendation - Transaction)** dataset, a comprehensive collection of real-world investment data.

| Dataset Component | Description | key Features Used |
| :--- | :--- | :--- |
| **Transactions** | Historical trade logs (Buy/Sell) | `customerID`, `ISIN`, `transactionType`, `timestamp` |
| **Asset Info** | Metadata for tradable assets | `assetCategory`, `sector`, `industry`, `marketID` |
| **Customer Info** | Investor profiles & demographics | `riskLevel`, `investmentCapacity`, `customerType` |
| **Limit Prices** | Historical price & performance | `profitability`, `priceMaxDate` |

*Data Preprocessing*: Transactions are filtered for "Buy" signals only, enforcing a positive-preference assumptions. Temporal sorting ensures strict causality validation (no look-ahead bias).

## Methodology: The 3-Pillar Hybrid Engine

### 1. Collaborative Filtering (The "Wisdom of Crowds")
*   **Technique**: Matrix Factorization using **Truncated SVD** (Singular Value Decomposition).
*   **Logic**: Decomposes the sparse User-Item interaction matrix into low-rank latent factors.
*   **Why**: Captures implicit relationships between users. If User A and User B have similar trading histories, SVD infers User A might like assets buying by User B, even if they have no explicit features in common.

### 2. Content-Based Filtering (The "Fundamentalist")
*   **Technique**: **Cosine Similarity** on One-Hot Encoded Features.
*   **Logic**: Constructs a "User Profile Vector" by averaging the features (Sector, Industry, Profitability) of assets they have previously purchased. Computes similarity between this profile and all candidate assets.
*   **Why**: Ensures recommendations align with the user's specific interests (e.g., a "Tech Sector" investor gets more Tech recommendations). Solves the cold-start problem for users with unique tastes.

### 3. Demographic Scoring (The "Risk Manager")
*   **Technique**: **Heuristic Weighted Matching**.
*   **Logic**: Maps user questionnaire responses (Risk Appetite, Investment Horizon, Capital) to asset risk categories.
*   **Why**: Acts as a safety layer. A "Conservative" investor is mathematically penalized for high-volatility asset classes, ensuring suitability compliance.

### 4. Hybrid Ensemble
The final recommendation score $S_{final}$ is a weighted linear combination:
$$ S_{final} = w_{cf} \cdot S_{cf} + w_{cb} \cdot S_{cb} + w_{demo} \cdot S_{demo} $$
*Weights ($w$) are dynamically adjustable in the dashboard to analyze component impact.*

## Performance & Evaluation
We employ a rigorous **Leave-One-Out (LOO)** cross-validation strategy:
*   **Train**: All transactions *except* the last one for each user.
*   **Test**: The most recent transaction for each user.

### Key Metrics
| Metric | Purpose | Interpretation |
| :--- | :--- | :--- |
| **RMSE** | Accuracy | Lower is better. Measures deviation of predicted ratings from actual. |
| **Precision@K** | Relevance | % of top-K recommendations that were actually purchased. |
| **Recall@K** | Coverage | Ability of the model to find the elusive "correct" asset in the top K. |
| **nDCG@K** | Ranking Quality | Weighted relevance. Rewards placing the correct asset higher up the list. |
| **ROI (Top-K)** | Financial Impact | Average profitability of the recommended portfolio. |

## Tech Stack
*   **Core Logic**: Python, Pandas, NumPy, Scikit-learn
*   **Interface**: Streamlit (Reactive Web Dashboard)
*   **Modeling**: TruncatedSVD, Cosine Similarity
*   **Validation**: Time-series split, Custom Ranking Metrics

## Project Structure
```text
├── app.py                  # Main Streamlit Application (Dashboard & Logic)
├── FAR-Trans-Data/         # Dataset Directory
│   ├── asset_information.csv
│   ├── customer_information.csv
│   ├── transactions.csv
│   └── ...
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation
```

## Usage Guide
1.  **Launch the Dashboard**: Run `streamlit run app.py`.
2.  **Configuration**: Use the sidebar to select a `Customer ID` and adjust `Algorithm Weights`.
3.  **Risk Profiling**: Navigate to "Risk Profile" tab to simulate a new investor questionnaire response.
4.  **Generate**: Click "Generate Recommendations" to run the hybrid engine in real-time.
5.  **Analyze**: View "Model Performance" tab for live LOO validation metrics.

## Conclusion
The **FAR-Trans Recommender** demonstrates that financial advice need not be a "black box." By explicitly combining latent behavioral signals with fundamental constraints and risk profiling, we achieve a system that is both **mathematically robust** and **financially responsible**. This architecture serves as a blueprint for next-generation Robo-Advisors.

## Keywords
Recommender Systems, Fintech, Collaborative Filtering, SVD, Content-Based Filtering, Hybrid AI, Wealth Management, Python, Streamlit, Scikit-learn, Behavioral Finance, Cold Start, Matrix Factorization.

---
*Citation*:
> Sanz-Cruzado, J., Droukas, N., & McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation.* IJCAI-2024 Workshop on Recommender Systems in Finance.