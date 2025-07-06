import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Required Files
# ----------------------------
try:
    # Load all necessary files
    rfm = pd.read_csv("D:/Mini_project_4/rfm.csv", index_col=0)
    item_sim_df = pd.read_csv("D:/Mini_project_4/item_similarity.csv", index_col=0)

    # ✅ Force product codes as strings in lookup
    product_lookup_df = pd.read_csv("D:/Mini_project_4/product_lookup.csv", index_col=0)
    product_lookup_df.index = product_lookup_df.index.astype(str)  # 🔄 This is critical!
    product_lookup = product_lookup_df["Description"].to_dict()

    # Load scaler & model
    scaler = joblib.load("D:/Mini_project_4/scaler.pkl")
    kmeans = joblib.load("D:/Mini_project_4/kmeans.pkl")

except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# ----------------------------
# Setup Reverse Lookup
# ----------------------------
# Product name ➜ StockCode
name_to_code = {v.strip().upper(): k for k, v in product_lookup.items()}

# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="wide")
st.title("🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendation")

tab1, tab2 = st.tabs(["📦 Product Recommendation", "👥 Customer Segmentation"])

# ----------------------------
# TAB 1: Product Recommendation (by Product Name)
# ----------------------------
with tab1:
    st.subheader("🔍 Product Recommender")

    # Input: Product Name
    input_name = st.text_input("Enter Product Name").strip().upper()

    # Recommendation Logic
    def recommend_by_product_name(name, top_n=5):
        code = name_to_code.get(name)
        if not code or code not in item_sim_df.columns:
            return []
        sim_codes = item_sim_df[code].sort_values(ascending=False).iloc[1:top_n+1].index.tolist()
        return [(c, product_lookup.get(str(c), "Name not available")) for c in sim_codes]

    if st.button("📥 Recommend"):
        if input_name in name_to_code:
            recommendations = recommend_by_product_name(input_name)
            if recommendations:
                st.subheader("📋 Top 5 Recommended Products:")
                for i, (code, name) in enumerate(recommendations, 1):
                    display_name = name if pd.notna(name) and name.strip() else "❓ Unknown Product Name"
                    st.markdown(f"**{i}. {display_name}**  \n*Product Code:* `{code}`")
            else:
                st.warning("⚠️ No recommendations found.")
        else:
            st.error("❌ Product name not found. Please check spelling.")

# ----------------------------
# TAB 2: Customer Segmentation
# ----------------------------
with tab2:
    st.subheader("📊 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary Value (total spend)", min_value=0.0)

    segment_labels = {
        0: "High-Value Shopper",
        1: "Regular Shopper",
        2: "Occasional Shopper",
        3: "At-Risk Shopper"
    }

    if st.button("📌 Predict Segment"):
        try:
            scaled = scaler.transform([[recency, frequency, monetary]])
            cluster = kmeans.predict(scaled)[0]
            label = segment_labels.get(cluster, "Unknown")
            st.success(f"📍 Cluster: {cluster}")
            st.info(f"This customer belongs to: **{label}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
