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

# Apply custom color theme inspired by your image
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #0B0C10;
    background-image: linear-gradient(180deg, #0B0C10 0%, #1F2833 100%);
    color: #C5C6C7;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1F2833;
    color: #C5C6C7;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #66FCF1 !important;
    font-weight: 700;
}

/* Subheadings and highlights */
h6, p, label, span {
    color: #C5C6C7 !important;
}

/* Buttons */
.stButton>button {
    background-color: #66FCF1;
    color: #0B0C10;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #45A29E;
    color: white;
}

/* Sidebar titles and radio buttons */
[data-testid="stSidebar"] .css-1d391kg, .css-16idsys, .css-1v3fvcr {
    color: #66FCF1 !important;
}

/* Tables */
table {
    color: #C5C6C7 !important;
    background-color: #1F2833 !important;
    border-radius: 10px;
}

/* Footer */
footer {
    color: #C5C6C7;
}
</style>
""", unsafe_allow_html=True)

# HEADER WITH COLLEGE LOGO AND TITLE
# -----------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("college_logo.png", width=110)  # 🔸 optional logo
with col2:
    st.markdown("""
        <style>
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-10px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .main-title {
            font-size: 56px;
            font-weight: 800;
            text-align: center;
            color: #66FCF1;
            text-shadow: 0 0 15px #45A29E, 0 0 35px #66FCF1, 0 0 55px #66FCF1;
            letter-spacing: 1px;
            animation: fadeIn 1.5s ease-in-out;
        }

        .subtitle {
            font-size: 22px;
            color: #45A29E;
            text-align: center;
            margin-top: -10px;
            animation: fadeIn 2s ease-in-out;
        }

        .team {
            text-align: center;
            color: #C5C6C7;
            margin-top: 20px;
            font-size: 16px;
            animation: fadeIn 2.5s ease-in-out;
        }

        .team span {
            display: inline-block;
            background: #1F2833;
            color: #66FCF1;
            padding: 6px 12px;
            border-radius: 12px;
            margin: 4px;
            font-weight: 600;
            box-shadow: 0 0 8px #45A29E;
        }
        </style>

        <div style="text-align:center; margin-bottom:30px;">
            <h1 class="main-title">Customer Segmentation</h1>
            <h3 class="subtitle">using K-Means Clustering</h3>
        </div>
    """, unsafe_allow_html=True)
    # --- Neon Sidebar Styling ---
st.markdown("""
<style>
/* Sidebar overall background */
section[data-testid="stSidebar"] {
    background-color: #1F2833;
    color: #C5C6C7;
    border-right: 2px solid #45A29E;
    box-shadow: 0 0 15px rgba(102,252,241,0.2);
}

/* Sidebar title */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #66FCF1 !important;
    text-shadow: 0 0 8px #45A29E;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-weight: 600;
    color: #C5C6C7;
    transition: all 0.2s ease-in-out;
}

/* Sidebar hover + selected states */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    color: #66FCF1;
    text-shadow: 0 0 10px #45A29E;
    transform: translateX(3px);
}
[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div > label {
    color: #66FCF1 !important;
    text-shadow: 0 0 10px #66FCF1, 0 0 20px #45A29E;
}

/* Sidebar icons (emoji-style) */
[data-testid="stSidebar"] div[role="radiogroup"] svg {
    fill: #66FCF1 !important;
}

/* Divider line */
section[data-testid="stSidebar"] hr {
    border: 1px solid #45A29E;
    opacity: 0.5;
}

/* Sidebar footer text */
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] caption {
    color: #C5C6C7;
    text-align: center;
    font-size: 13px;
    margin-top: 20px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)




# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.markdown("<h3 style='color:#66FCF1;'>📍 Navigation</h3>", unsafe_allow_html=True)
menu = st.sidebar.radio("", ["🏠 Home", "📂 Dataset", "📊 Clustering Results", "🧮 Predict New Customer"])


# -------------------------------
# Data Preprocessing
# -------------------------------
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("📂 Add or Use Dataset")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset uploaded successfully!")
else:
    df = pd.read_csv("dataset.csv")
    st.sidebar.info("Using default dataset (dataset.csv)")

df_proc = df.copy()
le = LabelEncoder()
df_proc["Gender_enc"] = le.fit_transform(df_proc["Gender"])
X = df_proc[["Gender_enc", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar: Dynamic K selection
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
k_value = st.sidebar.slider("🔢 Select number of clusters (k)", 2, 6, 5)

# Apply K-Means dynamically
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

# Update PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# Persona naming system for different k values
# --------------------------------------------------
if k_value == 2:
    persona_names = {
        0: "💎 Premium Customers",
        1: "🛒 Regular Customers"
    }

elif k_value == 3:
    persona_names = {
        0: "💎 High Spenders",
        1: "🧺 Moderate Buyers",
        2: "💰 Budget Savers"
    }

elif k_value == 4:
    persona_names = {
        0: "💎 Luxury Shoppers",
        1: "🧠 Smart Spenders",
        2: "💼 Working Class",
        3: "🎯 Young Spenders"
    }

elif k_value == 5:
    persona_names = {
        0: "💎 Luxury Spenders",
        1: "🛒 Budget Shoppers",
        2: "🎯 Average Customers",
        3: "💼 Young Professionals",
        4: "🏖️ High Income, Low Spending"
    }

elif k_value == 6:
    persona_names = {
        0: "💎 Affluent Shoppers",
        1: "🛒 Budget Conscious",
        2: "🎯 Average Spenders",
        3: "💼 Career Starters",
        4: "🏖️ Wealthy Minimalists",
        5: "📈 Growth Buyers"
    }
# Map persona names
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
    st.caption("📁 Currently using: **Custom uploaded dataset**" if uploaded_file else "📁 Currently using: **Default dataset.csv**")
    st.dataframe(df.head(10))
    st.write("### 📊 Basic Statistics")
    st.dataframe(df.describe())
# -------------------------------
# Clustering Results Page
# -------------------------------
elif menu == "📊 Clustering Results":
    st.title("📊 Clustering Results & Insights")

    # Compute Elbow Method
    inertias = []
    K = range(2, 7)
    for k in K:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        inertias.append(kmeans_test.inertia_)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Elbow Method")
        fig1, ax1 = plt.subplots(figsize=(3.8, 2.8), dpi=100)
        ax1.plot(K, inertias, "-o", markersize=5, linewidth=2, color="#000000")
        ax1.set_xlabel("Number of Clusters (k)", color="#000000", fontsize=9)
        ax1.set_ylabel("Inertia", color="#000000", fontsize=9)
        ax1.set_title("Elbow Method", color="#000000", fontsize=11)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1, use_container_width=False)

    with col2:
        st.subheader("🎨 Cluster Visualization (2D PCA)")
        fig2, ax2 = plt.subplots(figsize=(3.8, 2.8), dpi=100)
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.8, s=35)
        ax2.set_xlabel("PCA 1", color="#000000", fontsize=9)
        ax2.set_ylabel("PCA 2", color="#000000", fontsize=9)
        ax2.set_title(f"Customer Segments (k={k_value})", color="#000000", fontsize=11)
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2, use_container_width=False)

    # Cluster Summary and Persona Section
    st.subheader("🧩 Cluster Summary with Personas")
    cluster_summary = df.groupby(["Cluster", "Persona"])[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
    st.dataframe(cluster_summary)

    st.markdown("### 👥 Persona Insights")

    if k_value == 2:
        st.info("""
        💎 **Premium Customers:** High income and high spending — valuable clients.  
        🛒 **Regular Customers:** Average income and spending — consistent buyers.
        """)

    elif k_value == 3:
        st.info("""
        💎 **High Spenders:** Wealthy customers with strong purchasing power.  
        🧺 **Moderate Buyers:** Balanced income and spending.  
        💰 **Budget Savers:** Low spending despite moderate income.
        """)

    elif k_value == 4:
        st.info("""
        💎 **Luxury Shoppers:** Prefer premium products and higher spending.  
        🧠 **Smart Spenders:** Optimize purchases and seek value.  
        💼 **Working Class:** Steady earners with moderate spending.  
        🎯 **Young Spenders:** Energetic customers spending on lifestyle.
        """)

    elif k_value == 5:
        st.info("""
        💎 **Luxury Spenders:** High income, high spending — premium customers.  
        🛒 **Budget Shoppers:** Moderate income, careful spending.  
        🎯 **Average Customers:** Balanced income and spending behavior.  
        💼 **Young Professionals:** Low income but higher spending scores.  
        🏖️ **High Income, Low Spending:** Wealthy customers who spend conservatively.
        """)

    elif k_value == 6:
        st.info("""
        💎 **Affluent Shoppers:** Highest income, premium taste.  
        🛒 **Budget Conscious:** Prefer savings over spending.  
        🎯 **Average Spenders:** Balanced earners and spenders.  
        💼 **Career Starters:** Younger audience with growing income.  
        🏖️ **Wealthy Minimalists:** High earners with low spending.  
        📈 **Growth Buyers:** Increasing income and spending over time.
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
# --- Glowing Footer ---
st.markdown("""
<style>
.footer {
    text-align: center;
    padding: 20px 0;
    color: #C5C6C7;
    font-size: 15px;
    border-top: 1px solid #45A29E;
    margin-top: 50px;
    animation: ease-in-out infinite alternate;
}
</style>

<div class="footer">
    Developed by <b style='color:#66FCF1;'>Hanan</b>, <b style='color:#66FCF1;'>Dilber</b>, 
    <b style='color:#66FCF1;'>Dana</b>, <b style='color:#66FCF1;'>Abhayanth</b>, 
    and <b style='color:#66FCF1;'>Arya</b><br>
    <span style='font-size:14px;color:#45A29E;'>IML Project </span>
</div>
""", unsafe_allow_html=True)
