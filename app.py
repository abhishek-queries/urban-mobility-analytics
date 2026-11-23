import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Page Configuration
st.set_page_config(
    page_title="Urban Mobility & Ride-Hailing Analytics",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Urban Mobility & Ride-Hailing Analytics")
st.markdown(
    "**B.Tech AI & Data Science Minor Project | Poornima University, Jaipur**  \n"
    "End-to-end analytics pipeline covering data preprocessing, KPI metrics, demand trends, and K-Means customer segmentation."
)

# 1. Data Ingestion & Cleaning Module
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df["Booking ID"] = df["Booking ID"].astype(str).str.replace('"', "", regex=False)
    df["Customer ID"] = df["Customer ID"].astype(str).str.replace('"', "", regex=False)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.day_name()
    df["Month"] = df["Datetime"].dt.to_period("M").astype(str)
    return df

try:
    df = load_and_clean_data("rides_data.csv")
except FileNotFoundError:
    st.error("Dataset `rides_data.csv` not found. Please upload `rides_data.csv` to your repository.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filter Ride Records")
vehicle_options = ["All"] + sorted(df["Vehicle Type"].dropna().unique().tolist())
selected_vehicle = st.sidebar.selectbox("Select Vehicle Type", vehicle_options)

status_options = ["All"] + sorted(df["Booking Status"].dropna().unique().tolist())
selected_status = st.sidebar.selectbox("Select Booking Status", status_options)

filtered_df = df.copy()
if selected_vehicle != "All":
    filtered_df = filtered_df[filtered_df["Vehicle Type"] == selected_vehicle]
if selected_status != "All":
    filtered_df = filtered_df[filtered_df["Booking Status"] == selected_status]

completed_df = filtered_df[filtered_df["Booking Status"] == "Completed"]

# 2. KPI Metrics Row
st.subheader("📊 Key Performance Indicators (KPIs)")
col1, col2, col3, col4, col5 = st.columns(5)

total_bookings = len(filtered_df)
completed_count = len(completed_df)
total_revenue = completed_df["Booking Value"].sum()
avg_fare = completed_df["Booking Value"].mean() if completed_count > 0 else 0.0
avg_distance = completed_df["Ride Distance"].mean() if completed_count > 0 else 0.0

col1.metric("Total Bookings", f"{total_bookings:,}")
col2.metric("Completed Rides", f"{completed_count:,}")
col3.metric("Total Revenue (₹)", f"₹{total_revenue:,.0f}")
col4.metric("Avg Fare (₹)", f"₹{avg_fare:,.2f}")
col5.metric("Avg Distance (km)", f"{avg_distance:,.2f} km")

st.divider()

# Tabs for Structured Exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand & Revenue Trends",
    "📍 Locations & Vehicle Mix",
    "🤖 K-Means Customer Segmentation",
    "🗂️ Cleaned Data Preview"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        hourly = filtered_df.groupby("Hour").size().reset_index(name="Rides")
        fig_hour = px.line(
            hourly, x="Hour", y="Rides", markers=True,
            title="Hourly Ride Demand Pattern (24-Hour Cycle)",
            labels={"Hour": "Hour of Day (0–23)", "Rides": "Number of Bookings"}
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with c2:
        status_counts = filtered_df["Booking Status"].value_counts().reset_index()
        status_counts.columns = ["Booking Status", "Count"]
        fig_status = px.pie(
            status_counts, names="Booking Status", values="Count", hole=0.4,
            title="Booking Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    monthly_rev = completed_df.groupby("Month")["Booking Value"].sum().reset_index()
    fig_month = px.bar(
        monthly_rev, x="Month", y="Booking Value",
        title="Monthly Completed Ride Revenue (₹)",
        labels={"Booking Value": "Revenue (₹)"}
    )
    st.plotly_chart(fig_month, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        veh_perf = completed_df.groupby("Vehicle Type").agg(
            Completed_Rides=("Booking ID", "count"),
            Total_Revenue=("Booking Value", "sum"),
            Avg_Fare=("Booking Value", "mean")
        ).reset_index().sort_values("Total_Revenue", ascending=False)

        fig_veh = px.bar(
            veh_perf, x="Vehicle Type", y="Total_Revenue", color="Vehicle Type",
            title="Revenue by Vehicle Category (₹)",
            labels={"Total_Revenue": "Revenue (₹)"}
        )
        st.plotly_chart(fig_veh, use_container_width=True)

    with c2:
        top_pickups = filtered_df["Pickup Location"].value_counts().head(10).reset_index()
        top_pickups.columns = ["Pickup Location", "Bookings"]
        fig_loc = px.bar(
            top_pickups, x="Bookings", y="Pickup Location", orientation="h",
            title="Top 10 High-Demand Pickup Hotspots"
        )
        fig_loc.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_loc, use_container_width=True)

    st.dataframe(veh_perf, use_container_width=True)

with tab3:
    st.subheader("Customer Behavioral Clustering (K-Means)")
    st.markdown(
        "Aggregates completed trips per `Customer ID` using **Total Rides**, **Total Spend (₹)**, "
        "**Average Distance (km)**, and **Average Customer Rating**, then applies K-Means clustering."
    )

    k_clusters = st.slider("Select Number of Customer Clusters (K)", min_value=2, max_value=5, value=3)

    cust_df = df[df["Booking Status"] == "Completed"].groupby("Customer ID").agg(
        Total_Rides=("Booking ID", "count"),
        Total_Spend=("Booking Value", "sum"),
        Avg_Distance=("Ride Distance", "mean"),
        Avg_Rating=("Customer Rating", "mean")
    ).dropna().reset_index()

    features = ["Total_Rides", "Total_Spend", "Avg_Distance", "Avg_Rating"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cust_df[features])

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cust_df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=min(5000, len(X_scaled)), replace=False)
    sil_score = silhouette_score(X_scaled[sample_idx], cust_df["Cluster"].iloc[sample_idx])

    st.info(f"**Model Evaluation — Silhouette Score (sampled):** `{sil_score:.3f}` | **Customers Segmented:** `{len(cust_df):,}`")

    c1, c2 = st.columns(2)
    with c1:
        cluster_summary = cust_df.groupby("Cluster")[features].mean().round(2).reset_index()
        cluster_summary["Customer_Count"] = cust_df.groupby("Cluster").size().values
        st.markdown("**Cluster Centroid Summary (Averages)**")
        st.dataframe(cluster_summary, use_container_width=True)

    with c2:
        plot_sample = cust_df.sample(n=min(3000, len(cust_df)), random_state=42)
        fig_cluster = px.scatter(
            plot_sample, x="Avg_Distance", y="Total_Spend", color="Cluster",
            hover_data=["Total_Rides", "Avg_Rating"],
            title="Customer Segments: Avg Distance vs. Total Spend (Sampled)"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

with tab4:
    st.subheader("Sample Preprocessed Records")
    st.dataframe(filtered_df.head(100), use_container_width=True)
