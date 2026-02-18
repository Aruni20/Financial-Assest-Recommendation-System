import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
@st.cache_data
def load_data():
    # Load the CSV files (assumes UTF-8 encoding)
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    limit_prices_df = pd.read_csv("FAR-Trans-Data/limit_prices.csv")
    
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(transactions_df):
    # Only "Buy" as positive signal
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    # Sort by timestamp so that .tail(1) is most recent
    buys['timestamp'] = pd.to_datetime(buys.timestamp)
    buys = buys.sort_values('timestamp')
    return buys

def leave_one_out_split(buys):
    """
    Optimized leave-one-out split using vectorized operations.
    For each user, hold out their last-buy as test, rest as train.
    """
    # Ensure data is sorted
    buys = buys.sort_values(['customerID', 'timestamp'])
    
    # Identify the last transaction for each user
    # tail(1) gives the last row for each group
    last_idx = buys.groupby('customerID').tail(1).index
    
    # Test set is the last transaction
    test_df = buys.loc[last_idx]
    
    # Train set is everything else (drop rows where index is in last_idx)
    train_df = buys.drop(last_idx)
    
    return train_df, test_df

@st.cache_data
def build_rating_matrix(train_df):
    rating_df = train_df.groupby(['customerID','ISIN']).size().reset_index(name='count')
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='count').fillna(0)
    
    return rating_matrix, rating_df

########################################
# 2. COLLABORATIVE FILTERING COMPONENT
########################################
@st.cache_data
def matrix_factorization(rating_matrix, n_components=5):
    # Perform low-rank approximation with TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(rating_matrix)
    V = svd.components_.T  # shape: (num_assets, n_components)
    
    pred_ratings = np.dot(U, V.T)
    pred_df = pd.DataFrame(pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns)
    return pred_df

########################################
# 3. CONTENT-BASED FILTERING COMPONENT
########################################
def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    """
    Simplified content-based filtering using:
      - Asset category and subcategory
      - Sector and industry information
      - Market information
      - Profitability metrics
    """
    # Step 1: Prepare asset features
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']].copy()
    
    # Merge profitability
    asset_features = asset_features.merge(
        limit_prices_df[['ISIN', 'profitability']], 
        on='ISIN', 
        how='left'
    )
    
    # Fill missing values with medians/modes
    asset_features['profitability'] = asset_features['profitability'].fillna(asset_features['profitability'].median())
    asset_features['sector'] = asset_features['sector'].fillna('Unknown')
    asset_features['industry'] = asset_features['industry'].fillna('Unknown')
    
    # One-hot encode categorical features
    feature_cols = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']
    encoded_features = pd.get_dummies(asset_features[feature_cols])
    
    # Add profitability as a feature
    encoded_features['profitability'] = asset_features['profitability']
    
    # Set ISIN as index
    encoded_features.index = asset_features['ISIN']
    
    # Step 2: Build user profile
    # Get user's assets and convert to list for proper filtering
    user_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique().tolist()
    # Filter to only include assets that exist in our features
    user_assets = [asset for asset in user_assets if asset in encoded_features.index]
    
    if len(user_assets) == 0:
        # Cold start: return neutral scores
        return pd.Series(0.5, index=encoded_features.index)
    
    # Calculate user profile as mean of their asset features
    user_profile = encoded_features.loc[user_assets].mean()
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(
        user_profile.values.reshape(1, -1),
        encoded_features.values
    )[0]
    
    # Create series with ISINs as index
    content_scores = pd.Series(similarity_scores, index=encoded_features.index)
    
    return content_scores

########################################
# 4. DEMOGRAPHIC-BASED COMPONENT
########################################
def demographic_score(customer_id, customer_df, asset_df):
    """
    Returns a score for each asset based on how well the assetCategory aligns with the customer's
    demographic profile, including risk level, investment capacity, and other factors.
    """
    # Simplify predicted labels to their base forms
    def normalize_label(label):
        if pd.isna(label) or label == "Not_Available":
            return None
        return label.replace("Predicted_", "")
    
    # Mappings to numeric values
    risk_map = {
        "Conservative": 1, "Income": 2, "Balanced": 3, "Aggressive": 4
    }

    cap_map = {
        "CAP_LT30K": 1,
        "CAP_30K_80K": 2,
        "CAP_80K_300K": 3,
        "CAP_GT300K": 4
    }

    # Get latest record per customer
    customer_df_sorted = customer_df.sort_values("timestamp").drop_duplicates("customerID", keep="last")
    user_info = customer_df_sorted[customer_df_sorted["customerID"] == customer_id]

    if user_info.empty:
        return pd.Series(0.5, index=asset_df["ISIN"])  # fallback if no info

    # Extract basic demographic info
    risk = normalize_label(user_info["riskLevel"].values[0])
    cap = normalize_label(user_info["investmentCapacity"].values[0])
    customer_type = user_info["customerType"].values[0]

    # If values are missing, return neutral scores
    if risk not in risk_map or cap not in cap_map:
        return pd.Series(0.5, index=asset_df["ISIN"])

    # Create a more comprehensive user vector
    user_vector = np.array([
        risk_map[risk],  # Risk tolerance
        cap_map[cap],    # Investment capacity
        1 if customer_type == "Premium" else 0,  # Premium customer flag
        1 if customer_type == "Professional" else 0,  # Professional flag
    ])

    # Create average demographic vector for each assetCategory
    asset_scores = []
    for cat in asset_df["assetCategory"].unique():
        assets_in_cat = asset_df[asset_df["assetCategory"] == cat]
        
        # Get all customers who have invested in this category
        demographics = customer_df.copy()
        demographics["riskLevel"] = demographics["riskLevel"].apply(normalize_label)
        demographics["investmentCapacity"] = demographics["investmentCapacity"].apply(normalize_label)
        demographics = demographics.dropna(subset=["riskLevel", "investmentCapacity"])
        demographics = demographics[
            demographics["riskLevel"].isin(risk_map) & 
            demographics["investmentCapacity"].isin(cap_map)
        ]

        if demographics.empty:
            avg_vector = np.array([2.5, 2.5, 0.5, 0.5])  # neutral default
        else:
            avg_vector = np.array([
                demographics["riskLevel"].map(risk_map).mean(),
                demographics["investmentCapacity"].map(cap_map).mean(),
                (demographics["customerType"] == "Premium").mean(),
                (demographics["customerType"] == "Professional").mean()
            ])

        # Calculate similarity using weighted Euclidean distance
        weights = np.array([0.4, 0.3, 0.2, 0.1])  # Weights for each feature
        sim = 1 - np.sqrt(np.sum(weights * (user_vector - avg_vector) ** 2)) / np.sqrt(np.sum(weights * np.array([3, 3, 1, 1]) ** 2))
        asset_scores.append((cat, sim))

    category_sim_map = dict(asset_scores)

    # Assign each asset a score based on its category
    scores = asset_df["assetCategory"].map(category_sim_map).fillna(0.5)
    return pd.Series(scores.values, index=asset_df["ISIN"])

########################################
# 5. HYBRID RECOMMENDATION COMBINING THE THREE COMPONENTS
########################################
def normalize_scores(s):
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return s

def hybrid_recommendation(customer_id, rating_matrix, pred_df, rating_df, asset_df, 
                          customer_df, limit_prices_df, weights, top_n):
    """
    Combines:
      - Collaborative filtering (CF) score from matrix factorization
      - Content-based (CB) score from asset features
      - Demographic (DEMO) score based on customer profile
    """
    # 1. Collaborative Filtering
    if customer_id in pred_df.index:
        cf_scores = pred_df.loc[customer_id]
    else:
        cf_scores = pd.Series(0, index=rating_matrix.columns)
    
    # 2. Content-based Scores
    content_scores = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    
    # 3. Demographic-based Scores
    demo_scores = demographic_score(customer_id, customer_df, asset_df)
    
    # Normalize each score component to [0,1]
    cf_norm = normalize_scores(cf_scores)
    cb_norm = normalize_scores(content_scores)
    demo_norm = normalize_scores(demo_scores)
    
    # Weighted hybrid score
    final_score = weights[0]*cf_norm + weights[1]*cb_norm + weights[2]*demo_norm
    
    # Exclude assets that the customer has already bought
    bought_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique() if not rating_df[rating_df['customerID'] == customer_id].empty else []
    final_score = final_score.drop(labels=bought_assets, errors='ignore')
    
    recommendations = final_score.sort_values(ascending=False).head(top_n)
    return recommendations

#############################
# 6. EVALUATION METRICS
#############################
def compute_rmse(pred_df, test_df):
    """Compute RMSE only for user-item pairs in test set."""
    if test_df.empty:
        return None
        
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = row['customerID'], row['ISIN']
        if (u in pred_df.index) and (i in pred_df.columns):
            y_true.append(1.0)  # held-out buy = implicit rating 1
            y_pred.append(pred_df.at[u,i])
    
    if not y_true:
        return None
        
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_recall_at_n(pred_func, train_df, test_df, rating_matrix, rating_df, asset_df, customer_df, limit_prices_df, weights, pred_ratings, N):
    """Compute precision and recall at N for each user in test set."""
    if test_df.empty:
        return None, None
        
    precisions, recalls = [], []
    valid_users = 0
    
    for _, row in test_df.iterrows():
        try:
            u, test_isin = row['customerID'], row['ISIN']
            
            # Skip if user has no training data
            if u not in rating_matrix.index:
                continue
                
            # Generate recommendations for u
            recs = pred_func(u, rating_matrix, pred_ratings, rating_df, asset_df, customer_df, limit_prices_df, weights, top_n=N)
            
            # Skip if no recommendations could be generated
            if recs is None or len(recs) == 0:
                continue
                
            # Check if test item is in recommendations
            hit = int(test_isin in recs.index)
            precisions.append(hit / N)
            recalls.append(hit)  # since there's only 1 held-out item
            valid_users += 1
            
        except Exception as e:
            print(f"Error processing user {u}: {str(e)}")
            continue
    
    if valid_users == 0:
        return None, None
        
    return np.mean(precisions), np.mean(recalls)

def process_questionnaire_responses(responses):
    """
    Process questionnaire responses to determine risk level and investment capacity.
    Returns a tuple of (risk_level, investment_capacity)
    """
    # Risk level determination based on key questions
    risk_questions = {
        'q16': 0.3,  # Risk appetite
        'q17': 0.3,  # Investment expectations
        'q18': 0.2,  # Focus on gains vs losses
        'q19': 0.2   # Reaction to 20% decline
    }
    
    risk_score = 0
    for q, weight in risk_questions.items():
        if q in responses:
            answer = responses[q]
            if q == 'q16':  # Risk appetite
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
            elif q == 'q17':  # Investment expectations
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
            elif q == 'q18':  # Focus on gains vs losses
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
            elif q == 'q19':  # Reaction to decline
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
    
    # Map risk score to risk level
    if risk_score >= 3.5:
        risk_level = "Aggressive"
    elif risk_score >= 2.5:
        risk_level = "Balanced"
    elif risk_score >= 1.5:
        risk_level = "Income"
    else:
        risk_level = "Conservative"
    
    # Investment capacity determination
    if 'q13' in responses:  # Amount of funds available to invest
        investment = responses['q13']
        if investment == 'a':
            investment_capacity = "CAP_GT300K"
        elif investment == 'b':
            investment_capacity = "CAP_80K_300K"
        elif investment == 'c':
            investment_capacity = "CAP_30K_80K"
        else:
            investment_capacity = "CAP_LT30K"
    else:
        investment_capacity = "CAP_LT30K"  # Default to lowest capacity
    
    return risk_level, investment_capacity

def update_customer_profile(customer_id, risk_level, investment_capacity, customer_df):
    """Update customer profile with new questionnaire responses"""
    new_row = pd.DataFrame({
        'customerID': [customer_id],
        'customerType': ['Mass'],  # Default type
        'riskLevel': [risk_level],
        'investmentCapacity': [investment_capacity],
        'lastQuestionnaireDate': [datetime.now().strftime('%Y-%m-%d')],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    
    # Append new row to customer_df
    updated_df = pd.concat([customer_df, new_row], ignore_index=True)
    return updated_df

def compute_roi_at_k(recommendations, limit_prices_df, k=10):
    """
    Compute Return on Investment (ROI) for top-k recommendations.
    ROI is calculated using the profitability metric from limit_prices_df.
    """
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Get profitability for recommended assets
    roi_values = limit_prices_df.set_index('ISIN')['profitability'].loc[top_k.index]
    
    # Calculate average ROI
    avg_roi = roi_values.mean()
    
    return avg_roi

def compute_ndcg_at_k(recommendations, test_df, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at k.
    Uses the test set transactions as relevance indicators.
    """
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Create relevance list (1 if item is in test set, 0 otherwise)
    relevance = [1 if isin in test_df['ISIN'].values else 0 for isin in top_k.index]
    
    # Calculate DCG
    dcg = 0
    for i, rel in enumerate(relevance):
        dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal case: all relevant items are at the top)
    idcg = 0
    num_relevant = sum(relevance)
    for i in range(min(num_relevant, k)):
        idcg += 1 / np.log2(i + 2)
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg

#############################
# 7. STREAMLIT APP
#############################
def main():
    st.set_page_config(
        page_title="FAR-Trans AI Advisor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a professional, cleaner look
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #0E1117;
            font-weight: 700;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: 600;
        }
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .author-box {
            padding: 15px;
            background-color: #ffffff;
            border-radius: 8px;
            border-left: 5px solid #00a8cc;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 1.1em;
            color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Header
    st.sidebar.title("⚙️ Configuration")
    
    # Main Header
    st.title("📈 FAR-Trans Asset Recommender")
    st.markdown("### Intelligent Financial Asset Recommendations")
    
    # Updated Author Information - CLEARLY VISIBLE
    st.sidebar.markdown(
        """
        <div style="font-size: 0.8em; color: #888; margin-top: 20px;">
            Developed by: <strong>Aruni Saxena</strong>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Load & preprocess
    with st.spinner("Loading dataset and initializing models..."):
        asset_df, customer_df, transactions_df, limit_prices_df = load_data()
        buys = preprocess_data(transactions_df)
        train_df, test_df = leave_one_out_split(buys)
        rating_matrix, rating_df = build_rating_matrix(train_df)
        
        # CF Model Pre-computation
        pred_ratings = matrix_factorization(rating_matrix, n_components=5)

    # Sidebar controls
    st.sidebar.header("User Selection")
    customer_list = list(rating_matrix.index)
    customer_id_input = st.sidebar.selectbox("Select Customer ID", customer_list)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Settings")
    N = st.sidebar.number_input("Top N Recommendations", min_value=1, max_value=50, value=10)

    # Component Weights with Expanders to keep sidebar clean
    with st.sidebar.expander("⚖️ Algorithm Weights", expanded=True):
        if 'weights' not in st.session_state:
            st.session_state.weights = [0.4, 0.3, 0.3]
        
        cf_weight = st.slider("Collaborative Filtering", 0.0, 1.0, st.session_state.weights[0], 0.1)
        cb_weight = st.slider("Content-Based", 0.0, 1.0, st.session_state.weights[1], 0.1)
        demo_weight = st.slider("Demographic", 0.0, 1.0, st.session_state.weights[2], 0.1)
        
        st.session_state.weights = [cf_weight, cb_weight, demo_weight]
        weights = tuple(st.session_state.weights)

    # Main Content Area using Tabs (Reorganized)
    tab_guide, tab_recs, tab_risk, tab_perf, tab_about = st.tabs(["📚 User Guide", "🚀 Dashboard", "📝 Risk Profile", "📊 Model Performance", "ℹ️ About System"])
    
    # --- TAB 1: USER GUIDE ---
    with tab_guide:
        st.markdown("## 📚 How to Use This System")
        st.info("Follow these steps to generate personalized financial advice.")
        
        st.markdown("""
        ### Step 1: Configuration (Sidebar)
        1.  **Select Customer ID**: Choose a customer profile from the dropdown menu in the sidebar. This simulates logging in as a specific user.
        2.  **Top N Recommendations**: Choose how many assets you want recommended (e.g., 10 or 20).
        3.  **Algorithm Weights**: Adjust the sliders to favor different recommendation strategies:
            *   **Collaborative Filtering**: Looks for users with similar transaction history.
            *   **Content-Based**: Recommends assets similar to what you've bought before (based on sector, industry, etc.).
            *   **Demographic**: Matches assets to your risk profile and investment capacity.

        ### Step 2: Risk Profiling (Optional)
        *   Go to the **📝 Risk Profile** tab.
        *   Answer the questionnaire to update your risk tolerance (e.g., Conservative vs. Aggressive).
        *   Click "Update Risk Profile". This will influence the 'Demographic' component of the recommendation.

        ### Step 3: Generate Recommendations
        *   Go to the **🚀 Dashboard** tab.
        *   Click the **✨ Generate Recommendations** button.
        *   Review the top recommended assets, their profitability, and current prices.

        ### Step 4: Analyze Performance
        *   Check the metrics (ROI, nDCG) on the Dashboard.
        *   Go to the **📊 Model Performance** tab to see technical evaluation metrics (RMSE, Precision/Recall) for the entire system based on your current settings.
        """)

    # --- TAB 2: DASHBOARD ---
    with tab_recs:
        st.subheader(f"Portfolio Analysis & Recommendations for {customer_id_input}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info("Click the button below to generate personalized asset recommendations based on the hybrid model.")
            if st.button("✨ Generate Recommendations", type="primary"):
                with st.spinner("Analyzing profile and market data..."):
                    # Get recommendations
                    recs = hybrid_recommendation(customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df, 
                                                 customer_df, limit_prices_df, weights, top_n=int(N))
                    
                    st.session_state['last_recs'] = recs
                    st.session_state['recs_generated'] = True

        if st.session_state.get('recs_generated'):
            recs = st.session_state['last_recs']
            
            # Metrics Row
            roi = compute_roi_at_k(recs, limit_prices_df, k=10)
            ndcg = compute_ndcg_at_k(recs, test_df, k=10)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Avg ROI (Top 10)", f"{roi:.2%}" if roi else "N/A", delta_color="normal")
            with m2:
                st.metric("nDCG@10 (Relevance)", f"{ndcg:.4f}" if ndcg else "N/A")
            with m3:
                st.metric("Assets Recommended", len(recs))

            st.markdown("### Top Recommended Assets")
            
            # Detailed Dataframe
            rec_details = pd.DataFrame({
                'Score': recs,
                'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index],
                'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index],
                'Subcategory': asset_df.set_index('ISIN')['assetSubCategory'].loc[recs.index],
                'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index],
                'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index],
                'Current Price': limit_prices_df.set_index('ISIN')['priceMaxDate'].loc[recs.index]
            })
            
            st.dataframe(
                rec_details.style.format({
                    'Score': '{:.4f}',
                    'Profitability': '{:.2%}',
                    'Current Price': '€{:.2f}'
                }).background_gradient(subset=['Score'], cmap='Blues'),
                use_container_width=True,
                height=400
            )

    # --- TAB 3: RISK PROFILE ---
    with tab_risk:
        st.markdown("### 🧬 Investor Risk DNA")
        
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.write("Please update your investment preferences to refine demographic scoring.")
            
            # Initialize session state for questionnaire
            if 'questionnaire_responses' not in st.session_state:
                st.session_state.questionnaire_responses = {}
            
            # Questions Data (Reusing the structure)
            questions = {
                'q16': "How would you rate your appetite for 'risk'?",
                'q17': "Which of the following sentences best fits your investment expectations?",
                'q18': "Are you more concerned with potential losses or potential gains?",
                'q19': "If your investment declines by 20%, what would you do?",
                'q13': "Investment Capital Available:",
                'q6': "Investment Knowledge Level:",
                'q7': "Investment Experience:",
                'q8': "Trading Frequency (Last 3 Years):"
            }
            
            options = {
                'q16': {'a': "High (Risk Taker)", 'b': "Moderate-High", 'c': "Moderate", 'd': "Low", 'e': "Very Low (Risk Averse)"},
                'q17': {'a': "High Risk / High Return", 'b': "Growth with some risk", 'c': "Steady Income / Balanced", 'd': "Stable Income / Low Risk", 'e': "Capital Preservation"},
                'q18': {'a': "Always Profits", 'b': "Mostly Profits", 'c': "Balanced", 'd': "Mostly Losses", 'e': "Always Losses"},
                'q19': {'a': "Buy More (Opportunity)", 'b': "Rebalance", 'c': "Hold / Do Nothing", 'd': "Sell Part", 'e': "Sell All"},
                'q13': {'a': "> €1M", 'b': "€300k - €1M", 'c': "€80k - €300k", 'd': "€30k - €80k", 'e': "< €30k"},
                'q6': {'a': "Low", 'b': "Average", 'c': "Good", 'd': "Expert"},
                'q7': {'a': "None/Minimal", 'b': "Moderate", 'c': "Significant", 'd': "Extensive"},
                'q8': {'a': "Rarely", 'b': "Occasional", 'c': "Often", 'd': "Very Often"}
            }

            for q_id, question in list(questions.items())[:4]: # Split questions
                st.markdown(f"**{question}**")
                st.session_state.questionnaire_responses[q_id] = st.radio(
                    "Select option:", list(options[q_id].keys()), 
                    format_func=lambda x: options[q_id][x], key=f"risk_{q_id}", label_visibility="collapsed"
                )
                st.caption("")

        with col_q2:
            st.write(" &nbsp;") # Spacer
            for q_id, question in list(questions.items())[4:]:
                st.markdown(f"**{question}**")
                st.session_state.questionnaire_responses[q_id] = st.radio(
                    "Select option:", list(options[q_id].keys()), 
                    format_func=lambda x: options[q_id][x], key=f"bg_{q_id}", label_visibility="collapsed"
                )
                st.caption("")
                
            if st.button("Update Risk Profile", type="primary"):
                risk_level, investment_capacity = process_questionnaire_responses(st.session_state.questionnaire_responses)
                customer_df = update_customer_profile(customer_id_input, risk_level, investment_capacity, customer_df)
                st.success(f"✅ Profile Updated: **{risk_level}** Investigator with **{investment_capacity}** capacity.")

    # --- TAB 4: PERFORMANCE METRICS ---
    with tab_perf:
        st.markdown("### 📊 System Performance Evaluation")
        st.write("These metrics evaluate the recommendation quality using Leave-One-Out validation on the historical transaction dataset.")
        
        if st.button("Run Full System Evaluation"):
            with st.spinner("Computing metrics across test set (this may take a moment)..."):
                try:
                    rmse = compute_rmse(pred_ratings, test_df)
                    precision, recall = precision_recall_at_n(
                        hybrid_recommendation, train_df, test_df,
                        rating_matrix, rating_df, asset_df, customer_df, limit_prices_df,
                        weights, pred_ratings, N
                    )
                    
                    col_p1, col_p2, col_p3 = st.columns(3)
                    
                    with col_p1:
                        st.metric("RMSE (Root Mean Square Error)", f"{rmse:.4f}" if rmse else "N/A", help="Lower is better. Measures accuracy of rating predictions.")
                    
                    with col_p2:
                        st.metric(f"Precision@{N}", f"{precision:.4f}" if precision else "N/A", help=f"Percentage of top-{N} recommendations that were actually relevant (bought).")
                    
                    with col_p3:
                        st.metric(f"Recall@{N}", f"{recall:.4f}" if recall else "N/A", help=f"Percentage of relevant items found in the top-{N} recommendations.")
                    
                    st.success("Evaluation Complete!")
                    
                except Exception as e:
                    st.error(f"Error computing detailed metrics: {str(e)}")
        else:
            st.info("Click the button above to run the evaluation suite. Note: This can be computationally intensive.")


    # --- TAB 5: ABOUT ---
    with tab_about:
        st.markdown("""
        ### About FAR-Trans Asset Recommender
        
        This system leverages the **FAR-Trans dataset** to provide intelligent financial asset recommendations.
        It uses a hybrid approach combining:
        
        1.  **Collaborative Filtering (CF)**: Matrix Factorization (SVD) to find similar transaction patterns.
        2.  **Content-Based Filtering (CB)**: Matching asset features (Sector, Industry, Profitability) to user history.
        3.  **Demographic Scoring**: Aligning user risk profiles with asset categories.
        
        ---
        **Citation:**
        > Sanz-Cruzado, J., Droukas, N., & McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation.* IJCAI-2024 Workshop on Recommender Systems in Finance.
        """)
        
if __name__ == '__main__':
    main()
