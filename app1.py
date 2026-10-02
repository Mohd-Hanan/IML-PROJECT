import io
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------
# Constants
# -------------------------------
NUMERIC_COLS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
REQUIRED_COLS = ["Gender"] + NUMERIC_COLS
K_RANGE = list(range(2, 7))
DEFAULT_CSV = "dataset.csv"

# (income level, spending level) -> (emoji, segment name, marketing idea)
SEGMENTS = {
    ("high", "high"): ("💎", "Premium Spenders",
                       "Reward loyalty with early access to new arrivals, VIP offers and premium bundles."),
    ("high", "mid"): ("🏡", "Comfortable Earners",
                      "Grow their spending with curated recommendations and limited-time upgrades."),
    ("high", "low"): ("🏖️", "Cautious High Earners",
                      "Untapped potential: build trust with quality guarantees, reviews and personal outreach."),
    ("mid", "high"): ("🛍️", "Engaged Spenders",
                      "Keep them engaged with loyalty points, referral rewards and bundle deals."),
    ("mid", "mid"): ("⚖️", "Balanced Customers",
                     "Steady core: seasonal promotions and cross-selling to increase basket size."),
    ("mid", "low"): ("🧮", "Careful Spenders",
                     "Lead with value: discounts, price-match promises and combo offers."),
    ("low", "high"): ("🎯", "Enthusiastic Spenders",
                      "They spend beyond their income: offer instalments and affordable trendy items, and watch for churn."),
    ("low", "mid"): ("💡", "Value Seekers",
                     "Promote good-value essentials and entry-level or student deals."),
    ("low", "low"): ("🛒", "Budget Savers",
                     "Use low-cost entry offers, clearance sales and free-shipping thresholds."),
}

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Theme and styling
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0B0C10;
    background-image: linear-gradient(180deg, #0B0C10 0%, #1F2833 100%);
    color: #C5C6C7;
}
h1, h2, h3, h4, h5 { color: #66FCF1 !important; font-weight: 700; }
h6, p, label, span { color: #C5C6C7 !important; }
.stButton>button {
    background-color: #66FCF1; color: #0B0C10;
    border-radius: 8px; border: none; font-weight: 600;
}
.stButton>button:hover { background-color: #45A29E; color: white; }
table { color: #C5C6C7 !important; background-color: #1F2833 !important; border-radius: 10px; }

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-10px);}
    to {opacity: 1; transform: translateY(0);}
}
.main-title {
    font-size: 56px; font-weight: 800; text-align: center; color: #66FCF1;
    text-shadow: 0 0 15px #45A29E, 0 0 35px #66FCF1, 0 0 55px #66FCF1;
    letter-spacing: 1px; animation: fadeIn 1.5s ease-in-out;
}
.subtitle {
    font-size: 22px; color: #45A29E; text-align: center;
    margin-top: -10px; animation: fadeIn 2s ease-in-out;
}

section[data-testid="stSidebar"] {
    background-color: #1F2833; color: #C5C6C7;
    border-right: 2px solid #45A29E;
    box-shadow: 0 0 15px rgba(102,252,241,0.2);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #66FCF1 !important; text-shadow: 0 0 8px #45A29E;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-weight: 600; color: #C5C6C7; transition: all 0.2s ease-in-out;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    color: #66FCF1; text-shadow: 0 0 10px #45A29E; transform: translateX(3px);
}
section[data-testid="stSidebar"] hr { border: 1px solid #45A29E; opacity: 0.5; }

.footer {
    text-align: center; padding: 20px 0; color: #C5C6C7; font-size: 15px;
    border-top: 1px solid #45A29E; margin-top: 50px;
}
</style>

<div style="text-align:center; margin-bottom:30px;">
    <h1 class="main-title">Customer Segmentation</h1>
    <h3 class="subtitle">using K-Means Clustering</h3>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# Data helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes, default_csv_mtime):
    """Read the uploaded CSV, or the bundled dataset.csv when nothing is uploaded.

    default_csv_mtime is only there so the cache refreshes when dataset.csv changes.
    """
    if file_bytes is None:
        return pd.read_csv(DEFAULT_CSV)
    return pd.read_csv(io.BytesIO(file_bytes))


def clean_data(raw):
    """Validate columns, coerce numbers and drop unusable rows.

    Returns (dataframe, rows_dropped, error_message). error_message is None on success.
    """
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        return None, 0, "Your file is missing column(s): " + ", ".join(f"`{c}`" for c in missing)

    df = raw.copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    dropped = before - len(df)

    if len(df) < 10:
        return None, dropped, "At least 10 valid customer rows are needed to find segments."
    return df, dropped, None


@st.cache_data(show_spinner=False)
def k_diagnostics(X_scaled):
    """Inertia and silhouette score for every k in K_RANGE (cached, so it runs once per dataset)."""
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return inertias, silhouettes


def level(z, cut=0.5):
    return "high" if z > cut else "low" if z < -cut else "mid"


def build_personas(df):
    """Name each cluster from its actual averages, so labels stay correct for any k or dataset."""
    inc_col, spend_col = NUMERIC_COLS[1], NUMERIC_COLS[2]
    inc_mean, inc_std = df[inc_col].mean(), df[inc_col].std() or 1.0
    sp_mean, sp_std = df[spend_col].mean(), df[spend_col].std() or 1.0
    age_mean, age_std = df["Age"].mean(), df["Age"].std() or 1.0

    personas = {}
    for c, g in df.groupby("Cluster"):
        inc_lvl = level((g[inc_col].mean() - inc_mean) / inc_std)
        sp_lvl = level((g[spend_col].mean() - sp_mean) / sp_std)
        emoji, name, idea = SEGMENTS[(inc_lvl, sp_lvl)]
        age_z = (g["Age"].mean() - age_mean) / age_std
        age_tag = "younger" if age_z < -0.5 else "older" if age_z > 0.5 else "mid-age"
        personas[c] = {"emoji": emoji, "name": name, "idea": idea, "age_tag": age_tag}

    # Two clusters can land in the same segment: tell them apart by age, then by number
    counts = pd.Series([p["name"] for p in personas.values()]).value_counts()
    for c, p in personas.items():
        if counts[p["name"]] > 1:
            p["name"] = f'{p["name"]} ({p["age_tag"]})'
    counts = pd.Series([p["name"] for p in personas.values()]).value_counts()
    for c, p in personas.items():
        if counts[p["name"]] > 1:
            p["name"] = f'{p["name"]} #{c + 1}'

    for p in personas.values():
        p["label"] = f'{p["emoji"]} {p["name"]}'
    return personas


# -------------------------------
# Sidebar: navigation, data, k
# -------------------------------
st.sidebar.markdown("<h3>📍 Navigation</h3>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📂 Dataset", "📊 Clustering Results", "🧮 Predict New Customer"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("📂 Add or Use Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
st.sidebar.caption("Needed columns: " + ", ".join(REQUIRED_COLS))

try:
    raw_df = load_data(
        uploaded_file.getvalue() if uploaded_file is not None else None,
        os.path.getmtime(DEFAULT_CSV) if os.path.exists(DEFAULT_CSV) else 0,
    )
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

df, dropped_rows, problem = clean_data(raw_df)
if problem:
    st.error(problem)
    st.info("Fix the file and upload it again, or remove it to use the default dataset.")
    st.stop()

if uploaded_file is not None:
    st.sidebar.success("✅ Custom dataset uploaded successfully!")
else:
    st.sidebar.info(f"Using default dataset ({DEFAULT_CSV})")
if dropped_rows:
    st.sidebar.warning(f"{dropped_rows} row(s) with missing or invalid values were skipped.")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
k_value = st.sidebar.slider("🔢 Select number of clusters (k)", 2, 6, 5)

# -------------------------------
# Model
# -------------------------------
le = LabelEncoder()
gender_enc = le.fit_transform(df["Gender"])
X = np.column_stack([gender_enc, df[NUMERIC_COLS].to_numpy(dtype=float)])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

personas = build_personas(df)
df["Persona"] = df["Cluster"].map(lambda c: personas[c]["label"])
silhouette_avg = silhouette_score(X_scaled, clusters)

# -------------------------------
# Home
# -------------------------------
if menu == "🏠 Home":
    st.markdown("""
### 🎯 Project Overview
This project identifies **distinct customer segments** based on spending behavior.
Using **K-Means Clustering**, customers are grouped by *gender*, *age*, *annual income* and *spending score*.

### ⚙️ What you can do
- Explore the dataset and its statistics
- Pick the number of clusters and see how well they separate (elbow curve and silhouette score)
- Read a plain-language profile and a marketing idea for every segment
- Predict which segment a new customer belongs to
- Upload your own CSV and download the clustered result

### 🧠 Goal
Businesses can understand customer patterns and design **targeted marketing strategies**.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Average income (k$)", f"{df[NUMERIC_COLS[1]].mean():.1f}")
    c3.metric("Average spending score", f"{df[NUMERIC_COLS[2]].mean():.1f}")

# -------------------------------
# Dataset
# -------------------------------
elif menu == "📂 Dataset":
    st.title("📂 Dataset Preview")
    st.caption("📁 Currently using: **Custom uploaded dataset**" if uploaded_file else "📁 Currently using: **Default dataset.csv**")
    st.dataframe(df.drop(columns=["Cluster", "Persona"]).head(10))
    st.write("### 📊 Basic Statistics")
    st.dataframe(df[NUMERIC_COLS].describe())

# -------------------------------
# Clustering results
# -------------------------------
elif menu == "📊 Clustering Results":
    st.title("📊 Clustering Results & Insights")

    inertias, silhouettes = k_diagnostics(X_scaled)
    best_k = K_RANGE[int(np.argmax(silhouettes))]

    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette score (current k)", f"{silhouette_avg:.3f}", help="Higher is better (max = 1.0)")
    m2.metric("Best k by silhouette", best_k)
    m3.metric("Current k", k_value)
    if k_value == best_k:
        st.success(f"k = {k_value} has the best silhouette score of the values tested (2 to 6).")
    else:
        st.info(f"k = {best_k} scores best on silhouette. You can change k in the sidebar to compare.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Elbow Method")
        fig1, ax1 = plt.subplots(figsize=(4.2, 3.0), dpi=100)
        ax1.plot(K_RANGE, inertias, "-o", markersize=5, linewidth=2, color="#000000")
        ax1.axvline(k_value, color="#45A29E", linestyle="--", linewidth=1.5)
        ax1.set_xlabel("Number of Clusters (k)", fontsize=9)
        ax1.set_ylabel("Inertia", fontsize=9)
        ax1.set_xticks(K_RANGE)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
    with col2:
        st.subheader("📏 Silhouette by k")
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.0), dpi=100)
        ax2.bar(K_RANGE, silhouettes,
                color=["#45A29E" if k == k_value else "#B0B3B8" for k in K_RANGE])
        ax2.set_xlabel("Number of Clusters (k)", fontsize=9)
        ax2.set_ylabel("Silhouette score", fontsize=9)
        ax2.set_xticks(K_RANGE)
        ax2.grid(alpha=0.3, axis="y")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🎨 Cluster Visualization (2D PCA)")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        fig3, ax3 = plt.subplots(figsize=(4.2, 3.2), dpi=100)
        cmap = plt.get_cmap("tab10")
        for c in sorted(personas):
            mask = clusters == c
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], s=30, alpha=0.8,
                        color=cmap(c % 10), label=personas[c]["name"])
        ax3.set_xlabel("PCA 1", fontsize=9)
        ax3.set_ylabel("PCA 2", fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=6, loc="best")
        st.pyplot(fig3)
    with col4:
        st.subheader("👥 Segment Sizes")
        sizes = df["Cluster"].value_counts().sort_index()
        fig4, ax4 = plt.subplots(figsize=(4.2, 3.2), dpi=100)
        ax4.barh([personas[c]["name"] for c in sizes.index], sizes.values,
                 color=[cmap(c % 10) for c in sizes.index])
        ax4.invert_yaxis()
        ax4.set_xlabel("Customers", fontsize=9)
        ax4.tick_params(axis="y", labelsize=7)
        ax4.grid(alpha=0.3, axis="x")
        fig4.tight_layout()
        st.pyplot(fig4)

    st.subheader("🧩 Cluster Summary with Personas")
    summary = (
        df.groupby(["Cluster", "Persona"])
        .agg(Customers=("Age", "size"),
             **{"Avg Age": ("Age", "mean"),
                "Avg Income (k$)": ("Annual Income (k$)", "mean"),
                "Avg Spending Score": ("Spending Score (1-100)", "mean")})
        .round(1)
    )
    st.dataframe(summary)

    st.markdown("### 💾 Download Clustered Dataset")
    st.caption("Download the dataset grouped by clusters and personas to create targeted offers.")
    csv_data = df.sort_values(by="Cluster").to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Download Clustered Data (CSV)",
        data=csv_data,
        file_name=f"clustered_customers_k{k_value}.csv",
        mime="text/csv",
        help="Download the full dataset with assigned cluster and persona names.",
    )

    st.markdown("### 👥 Persona Insights")
    st.caption("Names and ideas are generated from each cluster's real averages, so they match the data you are looking at.")
    for c in sorted(personas):
        g = df[df["Cluster"] == c]
        p = personas[c]
        st.info(
            f"**{p['label']}** · {len(g)} customers ({len(g) / len(df):.0%})  \n"
            f"Average age {g['Age'].mean():.0f} · income {g[NUMERIC_COLS[1]].mean():.0f}k$ · "
            f"spending score {g[NUMERIC_COLS[2]].mean():.0f}  \n"
            f"💡 {p['idea']}"
        )

# -------------------------------
# Prediction
# -------------------------------
elif menu == "🧮 Predict New Customer":
    st.title("🧮 Predict a New Customer's Segment")

    def slider_bounds(series, lo_default, hi_default):
        lo, hi = int(np.floor(series.min())), int(np.ceil(series.max()))
        if lo >= hi:
            lo, hi = lo_default, hi_default
        return lo, hi, int(np.clip(round(series.median()), lo, hi))

    a_lo, a_hi, a_def = slider_bounds(df["Age"], 18, 70)
    i_lo, i_hi, i_def = slider_bounds(df[NUMERIC_COLS[1]], 10, 140)
    s_lo, s_hi, s_def = slider_bounds(df[NUMERIC_COLS[2]], 1, 100)

    age = st.slider("Age", a_lo, a_hi, a_def)
    income = st.slider("Annual Income (k$)", i_lo, i_hi, i_def)
    spending = st.slider("Spending Score (1-100)", s_lo, s_hi, s_def)
    gender = st.radio("Gender", list(le.classes_), horizontal=True)

    new_row = np.array([[le.transform([gender])[0], age, income, spending]], dtype=float)
    pred_cluster = int(kmeans.predict(scaler.transform(new_row))[0])
    p = personas[pred_cluster]
    g = df[df["Cluster"] == pred_cluster]

    st.success(f"✅ This customer belongs to **Cluster {pred_cluster}** → {p['label']}")
    st.info(
        f"Customers in this segment: {len(g)} · average age {g['Age'].mean():.0f} · "
        f"income {g[NUMERIC_COLS[1]].mean():.0f}k$ · spending score {g[NUMERIC_COLS[2]].mean():.0f}  \n"
        f"💡 {p['idea']}"
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
    Developed by <b style='color:#66FCF1;'>Hanan</b>, <b style='color:#66FCF1;'>Dilber</b>,
    <b style='color:#66FCF1;'>Dana</b>, <b style='color:#66FCF1;'>Abhayanth</b>,
    and <b style='color:#66FCF1;'>Arya</b><br>
    <span style='font-size:14px;color:#45A29E;'>IML Project </span>
</div>
""", unsafe_allow_html=True)
