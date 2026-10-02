<h1 align="center">🛒 Customer Segmentation</h1>

<p align="center">
  <b>Group customers by age, income and spending with K-Means, get a profile and marketing idea for each group, and predict the segment of a new customer.</b>
</p>

<p align="center">
  <a href="https://customer-segmentation-iml-project.streamlit.app/"><img src="https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live demo"/></a>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
</p>

<!-- Add a screenshot of the Clustering Results page here:
<p align="center"><img src="assets/clustering-results.png" width="800" alt="Clustering results page"/></p>
-->

## What it does

A Streamlit dashboard that splits customers into segments so a business can target offers at each group. You choose how many clusters to use, see how well they separate, read a plain-language profile of every segment, and check which segment a new customer would fall into.

**Try it live:** https://customer-segmentation-iml-project.streamlit.app/

## Features

- **Dynamic clustering:** pick k from 2 to 6 and the model re-clusters instantly
- **Cluster quality:** elbow curve, silhouette score for every k, and a suggestion for the best k
- **2D visualization:** PCA projection of the customer groups, plus segment sizes
- **Data-driven personas:** every segment is named and described from its real average income, spending and age, with a marketing idea attached (for example "Cautious High Earners")
- **Predict a new customer:** enter age, income, spending score and gender to get the segment and its profile
- **Bring your own data:** upload a CSV (columns are checked, bad rows are skipped with a warning) or use the included `dataset.csv`
- **Export:** download the dataset with cluster and persona columns added

## How it works

1. Gender is label-encoded. Gender, age, annual income and spending score are standardized with `StandardScaler`.
2. K-Means (`random_state=42`, `n_init=10`) groups the customers into the chosen number of clusters.
3. The elbow method and the silhouette score show how good that choice of k is.
4. Each cluster's average income and spending are compared with the whole dataset to name the segment.
5. PCA reduces the four features to two dimensions for plotting.
6. A new customer goes through the same scaler and is assigned to the nearest cluster.

## Run it locally

```bash
git clone https://github.com/Mohd-Hanan/IML-PROJECT.git
cd IML-PROJECT
pip install -r requirements.txt
streamlit run app1.py
```

## Use your own data

Upload a CSV from the sidebar. It needs these columns, spelled exactly like this:

`Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

## Project structure

```
IML-PROJECT/
├── app1.py            # Streamlit app: UI, clustering, personas, prediction
├── dataset.csv        # default customer dataset
└── requirements.txt
```

## Team

Group project by a team of five.
