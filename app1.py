import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# HEADER WITH COLLEGE LOGO AND TITLE
# -----------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("college_logo.png", width=110)  # 🔸 Add your logo file here
with col2:
    st.markdown("""
        <h1 style='font-size:36px; color:##ffffff; margin-bottom:0;'>🛍️ Customer Segmentation using K-Means Clustering</h1>
        <h4 style='color:#007bff; margin-top:4px;'> IML Project by Hanan,Dilber,Dana,Abhayanth,Arya</h4>
        <hr style="margin-top:10px; margin-bottom:20px;">
    """, unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Home", "📂 Dataset", "📊 Clustering Results", "🧮 Predict New Customer"]
)

# -------------------------------
# Data Preprocessing
# -------------------------------
df = pd.read_csv("dataset.csv")
df_proc = df.copy()
le = LabelEncoder()
df_proc["Gender_enc"] = le.fit_transform(df_proc["Gender"])
X = df_proc[["Gender_enc", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Define persona labels (you can modify these!)
persona_names = {
    0: "💎 Luxury Spenders",
    1: "🛒 Budget Shoppers",
    2: "🎯 Average Customers",
    3: "💼 Young Professionals",
    4: "🏖️ High Income, Low Spending"
}
df["Persona"] = df["Cluster"].map(persona_names)

# -------------------------------
# Home Page
# -------------------------------
if menu == "🏠 Home":
    st.markdown("""
    ### 🎯 Project Overview
    This project identifies **distinct customer segments** based on spending behavior.
    Using **K-Means Clustering**, customers are grouped by *Age*, *Annual Income*, and *Spending Score*.

    ### ⚙️ Tools Used
    - **Python**, **Pandas**, **Matplotlib**
    - **Scikit-learn** (Machine Learning)
    - **Streamlit** (Web App Framework)

    ### 🧠 Goal
    Businesses can understand customer patterns and design **targeted marketing strategies**.
    """)

# -------------------------------
# Dataset Page
# -------------------------------
elif menu == "📂 Dataset":
    st.title("📂 Dataset Preview")
    st.dataframe(df.head(10))
    st.write("### 📊 Basic Statistics")
    st.dataframe(df.describe())

# -------------------------------
# Clustering Results Page
# -------------------------------
elif menu == "📊 Clustering Results":
    st.title("📊 Clustering Results & Insights")

    # Elbow Method
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        inertias.append(kmeans_test.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(K, inertias, "-o")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method for Optimal k")
    st.subheader("📈 Elbow Method")
    st.pyplot(fig1)

    # PCA 2D Visualization
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.8)
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_title("Customer Segments (k=5)")
    plt.colorbar(scatter)
    st.subheader("🎨 Cluster Visualization (2D PCA)")
    st.pyplot(fig2)

    # Cluster Summary with Persona Names
    st.subheader("🧩 Cluster Summary with Personas")
    cluster_summary = df.groupby(["Cluster", "Persona"])[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
    st.dataframe(cluster_summary)

    # Persona Descriptions
    st.markdown("### 👥 Persona Insights")
    st.info("""
    💎 **Luxury Spenders:** High income, high spending — premium customers.  
    🛒 **Budget Shoppers:** Moderate income, careful spending.  
    🎯 **Average Customers:** Balanced income and spending behavior.  
    💼 **Young Professionals:** Low income but higher spending scores.  
    🏖️ **High Income, Low Spending:** Wealthy customers who spend conservatively.
    """)

# -------------------------------
# Prediction Page
# -------------------------------
elif menu == "🧮 Predict New Customer":
    st.title("🧮 Predict a New Customer's Segment")

    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Annual Income (k$)", 10, 140, 60)
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)
    gender = st.radio("Gender", ["Male", "Female"])

    gender_enc = 1 if gender == "Male" else 0
    new_data = scaler.transform([[gender_enc, age, income, spending]])
    pred_cluster = kmeans.predict(new_data)[0]
    persona = persona_names[pred_cluster]

    st.success(f"✅ This customer belongs to **Cluster {pred_cluster}** → {persona}")

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Developed by [Your Name] • IML Project 2025")
