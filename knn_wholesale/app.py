import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🟢",
    layout="wide"
)

# =====================================================
# WOW CSS (UNCHANGED – SAFE)
# =====================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #000000, #0f2027);
    color: silver;
    font-family: "Segoe UI", sans-serif;
}
.block-container {
    padding: 2.5rem 3rem;
}
h1 {
    color: gold;
    text-align: center;
    text-shadow: 0px 0px 12px gold;
}
h2, h3 {
    color: #00ff99;
    text-shadow: 0px 0px 6px #00ff99;
}
p { color: silver; font-size: 16px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000000, #1a1a1a);
    border-right: 3px solid gold;
}
section[data-testid="stSidebar"] label { color: gold !important; }
button {
    background: linear-gradient(45deg, red, gold) !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 12px !important;
}
.stPyplot {
    background-color: black;
    border: 2px solid gold;
    border-radius: 15px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE & DESCRIPTION
# =====================================================
st.markdown("<h1>🟢 Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)
st.info("👉 Discover hidden customer groups without predefined labels.")

# =====================================================
# DATA LOADING (BULLET-PROOF)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(BASE_DIR, "data", "Wholesale_customers_data.csv"),
    os.path.join(BASE_DIR, "Wholesale_customers_data.csv")
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.success(f"✅ Data loaded from: {path}")
        break

# -----------------------------------------------------
# If file not found → Upload CSV (FINAL SAFETY NET)
# -----------------------------------------------------
if df is None:
    st.warning("⚠️ Dataset not found. Please upload the CSV file below.")
    uploaded_file = st.file_uploader(
        "Upload Wholesale Customers CSV",
        type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.stop()

# =====================================================
# DATA PREVIEW
# =====================================================
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("📌 Clustering Controls")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
feature_2 = st.sidebar.selectbox("Select Feature 2", numeric_cols)

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)
random_state = st.sidebar.number_input("Random State", value=42)

run = st.sidebar.button("🟦 Run Clustering")

# =====================================================
# VALIDATION
# =====================================================
if feature_1 == feature_2:
    st.warning("⚠️ Please select **two different numerical features**.")
    st.stop()

# =====================================================
# RUN CLUSTERING
# =====================================================
if run:
    X = df[[feature_1, feature_2]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = labels

    # =================================================
    # VISUALIZATION
    # =================================================
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(k):
        ax.scatter(
            X[labels == i, 0],
            X[labels == i, 1],
            s=80,
            label=f"Cluster {i}"
        )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=300,
        c="black",
        edgecolors="gold",
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("K-Means Clustering Result")
    ax.legend()

    st.pyplot(fig)
    st.subheader("📋 Cluster Summary")

    summary = df.groupby("Cluster").agg(
        Count=(feature_1, "count"),
        **{f"Avg {feature_1}": (feature_1, "mean")},
        **{f"Avg {feature_2}": (feature_2, "mean")}
    ).reset_index()

    st.dataframe(summary)



    # =================================================
    # BUSINESS INTERPRETATION
    # =================================================
    st.subheader("💼 Business Interpretation")

    for i in range(k):
        st.write(
            f"🟢 **Cluster {i}:** Customers in this cluster show similar "
            f"purchasing behavior for **{feature_1}** and **{feature_2}**."
        )

    st.success(
        "📌 Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.warning("⬅️ Select inputs and click **Run Clustering** to start.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div style="text-align:center; color:silver; margin-top:20px;">
⚡ Built with Streamlit | K-Means | StandardScaler<br>
🎓 Academic Ready • 💼 Industry Ready
</div>
""", unsafe_allow_html=True)
