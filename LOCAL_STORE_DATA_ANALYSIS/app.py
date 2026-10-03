import streamlit as st
import pandas as pd
import numpy as np

from modules import id3
from modules import j48
from modules import naive_bayes
from modules import decision_tree
from modules import kmeans
from modules import apriori
from modules.data_loader import (
    load_clean_data,
    TARGET_OPTIONS,
    COL_STORE_TYPE,
    COL_EXPERIENCE,
    COL_PROBLEMS,
    COL_STOCK_ACTIONS,
    COL_STOCKOUT_LOSS,
    COL_DIGITAL_POS,
    COL_DEMAND_IMPACT,
    COL_STOCK_WASTE,
    COL_DEMAND_HELP,
    COL_QUICK_DELIVERY_IMPACT,
    COL_ADAPTATION,
    COL_PRICING_DIFFICULTY,
    COL_PRICING_STRATEGY,
    COL_SUPPLIER_DELAYS,
    COL_SUPPLIER_BACKUP,
    COL_RETENTION_SERVICES,
    COL_SOCIAL_MEDIA_SPIKES,
    COL_RESTOCK_SPEED,
    LABEL_STOCKOUT_LOSS,
    LABEL_DIGITAL_POS,
    LABEL_DEMAND_IMPACT,
    LABEL_SUPPLIER_DELAYS,
    LABEL_STOCK_WASTE,
    LABEL_QUICK_DELIVERY,
    LABEL_PRICING_DIFFICULTY,
    LABEL_RESTOCK_SPEED,
    LABEL_SOCIAL_MEDIA
)



# =========================================================
# PAGE CONFIGURATION & THEME SYSTEM
# =========================================================

st.set_page_config(
    page_title="Local Store Data Mining & Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit App CSS Styling with Curated Solid Teal & Complementary Terracotta/Coral Palette
# Solid Warm Beige Canvas (#FAF7F2), Crisp White Surfaces (#FFFFFF), Solid Mint-Sand Sidebar (#F2F8F7), Solid Teal (#0D9488), Terracotta Accent (#C2410C)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAF7F2;
        color: #1E293B;
    }
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1240px;
    }
    section[data-testid="stSidebar"] {
        background-color: #F0F6F5 !important;
        border-right: 1px solid #D5E5E2;
    }
    section[data-testid="stSidebar"] * {
        color: #134E4A !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #134E4A !important;
        font-weight: 750;
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 850;
        color: #0F766E;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
        text-transform: uppercase;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #5A7875;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #D5E5E2;
        padding-bottom: 8px;
    }
    label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p {
        color: #134E4A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #99F6E4 !important;
        color: #0F172A !important;
        border-radius: 6px;
    }
    div[data-baseweb="select"] * {
        color: #0F172A !important;
    }
    .insight-card {
        border-left: 4px solid #0D9488;
        background-color: #FFFFFF;
        border-top: 1px solid #D5E5E2;
        border-bottom: 1px solid #D5E5E2;
        border-right: 1px solid #D5E5E2;
        padding: 16px 20px;
        margin: 14px 0px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        line-height: 1.65;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(13, 148, 136, 0.06);
    }
    .viva-card {
        border: 1px solid #D5E5E2;
        border-left: 4px solid #0D9488;
        background-color: #FFFFFF;
        padding: 14px 18px;
        margin: 10px 0px;
        border-radius: 6px;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricValue"] {
        color: #0D9488 !important;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        color: #5A7875 !important;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #D5E5E2;
        border-radius: 6px;
    }
    button[data-baseweb="tab"] {
        color: #5A7875 !important;
        font-weight: 500;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0D9488 !important;
        border-bottom-color: #0D9488 !important;
        font-weight: 750;
    }
    hr {
        border-color: #D5E5E2 !important;
    }
    .stButton>button {
        background-color: #0D9488;
        color: #FFFFFF;
        border: 1px solid #0F766E;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        transition: background-color 0.2s ease, border-color 0.2s ease;
        box-shadow: none;
    }
    .stButton>button:hover {
        background-color: #0F766E;
        border-color: #115E59;
        color: #FFFFFF;
        box-shadow: none;
    }
    .stButton>button:active {
        background-color: #115E59;
        color: #FFFFFF;
    }
    /* Pinpoint Radio Button Circle Styling - Green/Teal fill ONLY inside the circle */
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background-color: transparent !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        background-color: #FFFFFF !important;
        border: 2px solid #94A3B8 !important;
        border-radius: 50% !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child,
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
        border-color: #0D9488 !important;
        background-color: #0D9488 !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child > div,
    div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child > div {
        background-color: #FFFFFF !important;
        border-radius: 50% !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:nth-child(2),
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {
        background-color: transparent !important;
        border: none !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:nth-child(2) *,
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child * {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="main-title">📊 Local Store Data Mining & Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retail Business Intelligence, Operational Risk & Predictive Analytics Platform</div>', unsafe_allow_html=True)



# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("### **Navigation**")

nav_option = st.sidebar.radio(
    "Go to",
    [
        "📊 1. Overview & Dataset",
        "📈 2. Exploratory Data Analysis (EDA)",
        "🌳 3. ID3 Decision Tree",
        "🌿 4. J48 Decision Tree",
        "🌲 5. Decision Tree (CART / Gini)",
        "🎯 6. Naive Bayes Classification",
        "📍 7. K-Means Clustering",
        "🛒 8. Apriori Association Rules"
    ]
)


df = load_clean_data()
TOTAL_STORES = len(df)


# =========================================================
# 1. OVERVIEW & DATASET
# =========================================================

if nav_option.startswith("📊 1.") or nav_option.startswith("1."):

    st.subheader("📊 Overview & Problem Statement")

    st.write(
        """
        This project analyzes survey data from **64 local brick-and-mortar stores** across **21 operational variables**.
        
        The objective is to analyze real-life retail challenges: **inventory planning, demand unpredictability, supplier delivery delays, lost sales due to unexpected stockouts, digital POS adoption, and competition from 10-minute quick-commerce delivery apps**, using standard data mining algorithms.
        """
    )

    st.markdown("### **Baseline Dataset Metrics**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stockout_rate = (df[COL_STOCKOUT_LOSS].value_counts().get("Yes", 0) / TOTAL_STORES) * 100
        st.metric("📦 Stockout Sales Loss", f"{stockout_rate:.1f}%", help="48 of 64 stores experienced lost sales due to unexpected stockouts.")

    with col2:
        pos_rate = (df[COL_DIGITAL_POS].value_counts().get("Yes", 0) / TOTAL_STORES) * 100
        st.metric("💳 Digital POS Adoption", f"{pos_rate:.1f}%", help="36 of 64 stores use a POS or inventory tracking app.")

    with col3:
        delayed_stores = df[COL_SUPPLIER_DELAYS].isin(["Daily", "Weekly", "Monthly"]).sum()
        delay_rate = (delayed_stores / TOTAL_STORES) * 100
        st.metric("⏳ Supplier Delivery Delays", f"{delay_rate:.1f}%", help="48 of 64 stores face regular supplier delivery delays.")

    with col4:
        qc_impacted = df[COL_QUICK_DELIVERY_IMPACT].apply(
            lambda x: 1 if ("Moderately" in str(x) or "Severely" in str(x) or "Slightly" in str(x)) else 0
        ).sum()
        qc_rate = (qc_impacted / TOTAL_STORES) * 100
        st.metric("⚡ Quick Commerce Threat", f"{qc_rate:.1f}%", help="51 of 64 stores reported sales/footfall drops from 10-min apps.")

    st.markdown("---")

    st.markdown("### **Analytical Framework**")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            **Classification Tasks (Supervised):**
            - **Selectable Real-Life Targets:**
              1. `Stockout Sales Loss (Inventory Risk)` — Yes / No
              2. `Digital POS Adoption (Technology Modernization)` — Yes / No
              3. `Quick Commerce Threat Level (Market Disruption)` — High Impact vs Low/No Impact
              4. `Profit Vulnerability to Demand Swings (Financial Risk)` — High Risk (4-5) vs Manageable (1-3)
              5. `Frequent Supplier Delivery Delays (Supply Chain Fragility)` — Frequent (Weekly/Daily) vs Reliable
            - **Models Applied:** ID3 (Entropy), J48 (C4.5-style), Decision Tree (Gini), Naive Bayes (Gaussian).
            - **Explanatory Features:** Store duration, POS usage, waste frequency, supplier delays, pricing pressure, restock lead times, and operational coping actions (leak-free matrix).
            """
        )

    with col_b:
        st.markdown(
            """
            **Unsupervised Data Mining Tasks:**
            - **Store Segmentation (K-Means):** Clusters stores based on 7 normalized operational friction metrics (Demand impact, pricing difficulty, supplier delays, stock spoilage, quick commerce drop, digital POS, and restocking lead time).
            - **Pattern Mining (Apriori):** Extracts association rules across multi-select problems, coping strategies, and outcomes using Support, Confidence, and Lift.
            """
        )



# =========================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

elif nav_option.startswith("📈 2.") or nav_option.startswith("2."):

    st.subheader("📈 Exploratory Data Analysis")

    tab_profile, tab_inventory, tab_demand, tab_suppliers, tab_tech, tab_competition = st.tabs([
        "🏪 Store Profile",
        "📦 Inventory & Stockouts",
        "📈 Demand & Trends",
        "🚚 Suppliers & Delays",
        "💻 Technology Adoption",
        "⚡ Competition & Quick Commerce"
    ])


    with tab_profile:
        col1, col2 = st.columns(2)
        with col1:
            type_counts = df[COL_STORE_TYPE].value_counts()
            st.markdown("**Store Category Distribution:**")
            st.dataframe(
                pd.DataFrame({
                    "Store Category": type_counts.index,
                    "Count": type_counts.values,
                    "Percentage (%)": (type_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )
            st.bar_chart(type_counts)
        with col2:
            exp_counts = df[COL_EXPERIENCE].value_counts()
            st.markdown("**Operating Duration / Experience:**")
            st.dataframe(
                pd.DataFrame({
                    "Experience": exp_counts.index,
                    "Count": exp_counts.values,
                    "Percentage (%)": (exp_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )
            st.bar_chart(exp_counts)

        st.markdown(
            """
            <div class="insight-card">
            <b>Summary:</b> Food/Cafe/Restaurant (29.7%), Clothing (25.0%), and Grocery (21.9%) make up <b>76.6%</b> of all stores. 
            75.0% of store owners have 3 years or less of operating experience (37.5% under 1 year, 37.5% between 1–3 years).
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_inventory:
        col1, col2 = st.columns(2)
        with col1:
            stockout_counts = df[COL_STOCKOUT_LOSS].value_counts()
            st.markdown("**" + LABEL_STOCKOUT_LOSS + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Status": stockout_counts.index,
                    "Store Count": stockout_counts.values,
                    "Percentage (%)": (stockout_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

            prob_items = [
                "Unpredictable changes in customer demand",
                "Supplier delays or inconsistent delivery",
                "Seasonal fluctuations and unexpected weather",
                "Lack of reliable historical sales data",
                "Inadequate software or inventory tracking tools"
            ]
            prob_counts = {p: df[COL_PROBLEMS].apply(lambda s: 1 if p in str(s) else 0).sum() for p in prob_items}
            prob_df = pd.DataFrame(list(prob_counts.items()), columns=["Inventory Problem", "Count"]).sort_values("Count", ascending=False)
            st.markdown("**Inventory & Demand Forecasting Problems:**")
            st.dataframe(prob_df, use_container_width=True)

        with col2:
            waste_counts = df[COL_STOCK_WASTE].value_counts()
            st.markdown("**" + LABEL_STOCK_WASTE + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Frequency": waste_counts.index,
                    "Count": waste_counts.values,
                    "Percentage (%)": (waste_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

            action_items = [
                "Maintain safety stock / buffer inventory",
                "Place emergency orders at higher supplier costs",
                "Run discount promotions to clear excess stock",
                "Offer substitute products to customers",
                "Manually audit stock levels more frequently"
            ]
            action_counts = {a: df[COL_STOCK_ACTIONS].apply(lambda s: 1 if a in str(s) else 0).sum() for a in action_items}
            action_df = pd.DataFrame(list(action_counts.items()), columns=["Stock Action", "Count"]).sort_values("Count", ascending=False)
            st.markdown("**Actions Taken During Stock Excess or Shortages:**")
            st.dataframe(action_df, use_container_width=True)

        st.markdown(
            """
            <div class="insight-card">
            <b>Summary:</b> <b>75.0% of stores (48 of 64)</b> lost sales because a product was unexpectedly out of stock. 
            The two most common inventory problems are <i>unpredictable customer demand</i> (64.1%) and <i>supplier delivery delays</i> (48.4%). 
            Stores primarily maintain buffer stock (68.8%) and place emergency orders at higher supplier costs (40.6%) to manage stock imbalances.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_demand:
        col1, col2 = st.columns(2)
        with col1:
            demand_impact_counts = df[COL_DEMAND_IMPACT].value_counts().sort_index()
            st.markdown("**" + LABEL_DEMAND_IMPACT + " (1=Low, 5=Severe):**")
            st.dataframe(
                pd.DataFrame({
                    "Rating (1-5)": demand_impact_counts.index,
                    "Count": demand_impact_counts.values,
                    "Percentage (%)": (demand_impact_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )
            st.bar_chart(demand_impact_counts)

            help_counts = df[COL_DEMAND_HELP].value_counts()
            st.markdown("**Preferred Demand Forecasting Solutions:**")
            st.dataframe(
                pd.DataFrame({
                    "Solution": help_counts.index,
                    "Count": help_counts.values,
                    "Percentage (%)": (help_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

        with col2:
            social_counts = df[COL_SOCIAL_MEDIA_SPIKES].value_counts()
            st.markdown("**" + LABEL_SOCIAL_MEDIA + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Social Trend Spikes": social_counts.index,
                    "Count": social_counts.values,
                    "Percentage (%)": (social_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

            restock_counts = df[COL_RESTOCK_SPEED].value_counts()
            st.markdown("**" + LABEL_RESTOCK_SPEED + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Restocking Speed": restock_counts.index,
                    "Count": restock_counts.values,
                    "Percentage (%)": (restock_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

        st.markdown(
            """
            <div class="insight-card">
            <b>Summary:</b> <b>76.6% of stores</b> rate profit sensitivity to demand swings as moderate-to-severe (3–5 rating). 
            <b>81.3% of stores (52 of 64)</b> have experienced viral social-media demand spikes, but only 28.1% can restock trending items within 1–2 days. 
            <b>54.7% of store owners stated that better supplier communication & reliability</b> would help them predict demand most.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_suppliers:
        col1, col2 = st.columns(2)
        with col1:
            delay_counts = df[COL_SUPPLIER_DELAYS].value_counts()
            st.markdown("**" + LABEL_SUPPLIER_DELAYS + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Delay Frequency": delay_counts.index,
                    "Count": delay_counts.values,
                    "Percentage (%)": (delay_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )
            st.bar_chart(delay_counts)

        with col2:
            backup_items = [
                "Inform customers to wait for the restock",
                "Switch to alternative brands or substitute products",
                "Buy from local wholesalers at a higher price",
                "Borrow stock from neighboring stores",
                "Source from a different city or region"
            ]
            backup_counts = {b: df[COL_SUPPLIER_BACKUP].apply(lambda s: 1 if b in str(s) else 0).sum() for b in backup_items}
            backup_df = pd.DataFrame(list(backup_counts.items()), columns=["Emergency Plan", "Count"]).sort_values("Count", ascending=False)
            st.markdown("**Emergency Plans During Supplier Shortages:**")
            st.dataframe(backup_df, use_container_width=True)

        st.markdown(
            """
            <div class="insight-card">
            <b>Summary:</b> <b>75.0% of stores (48 of 64)</b> experience regular supplier delivery delays (42.2% monthly, 28.1% weekly, 4.7% daily). 
            During shortages, 45.3% inform customers to wait, 42.2% switch to substitute brands, and 29.7% buy from local wholesalers at higher prices.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_tech:
        col1, col2 = st.columns(2)
        with col1:
            pos_counts = df[COL_DIGITAL_POS].value_counts()
            st.markdown("**" + LABEL_DIGITAL_POS + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Digital POS Adoption": pos_counts.index,
                    "Count": pos_counts.values,
                    "Percentage (%)": (pos_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

        with col2:
            st.markdown("**Cross-Tabulation: Digital POS Usage vs Stockout Sales Loss:**")
            ctab = pd.crosstab(
                df[COL_DIGITAL_POS],
                df[COL_STOCKOUT_LOSS],
                margins=True,
                margins_name="Total"
            )
            st.dataframe(ctab, use_container_width=True)

        st.markdown(
            """
            <div class="insight-card">
            <b>Summary:</b> <b>56.2% of stores (36 of 64)</b> have adopted digital POS systems. 
            However, <b>72.2% of digitized stores (26 of 36) still lost sales because a product was out of stock</b>, compared to 78.6% of non-digital stores (22 of 28). 
            Digital software records past sales but cannot prevent stockouts caused by external supplier delivery delays.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_competition:
        col1, col2 = st.columns(2)
        with col1:
            qc_counts = df[COL_QUICK_DELIVERY_IMPACT].value_counts()
            st.markdown("**" + LABEL_QUICK_DELIVERY + ":**")
            st.dataframe(
                pd.DataFrame({
                    "Impact Level": qc_counts.index,
                    "Count": qc_counts.values,
                    "Percentage (%)": (qc_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

            pricing_counts = df[COL_PRICING_DIFFICULTY].value_counts().sort_index()
            st.markdown("**" + LABEL_PRICING_DIFFICULTY + " (1=Easy, 5=Very Difficult):**")
            st.dataframe(
                pd.DataFrame({
                    "Difficulty Rating (1-5)": pricing_counts.index,
                    "Count": pricing_counts.values,
                    "Percentage (%)": (pricing_counts.values / TOTAL_STORES * 100).round(1)
                }),
                use_container_width=True
            )

        with col2:
            adapt_items = [
                "Offering home delivery via phone/WhatsApp orders",
                "Adopting digital payments (UPI) and modern billing",
                "Partnering directly with quick commerce or delivery platforms",
                "Enhancing personalized customer relationships and in-store experience",
                "Stocking fresh daily essentials (e.g., local bread, fresh milk) that platforms struggle to supply",
                "No specific strategy implemented yet"
            ]
            adapt_counts = {a: df[COL_ADAPTATION].apply(lambda s: 1 if a in str(s) else 0).sum() for a in adapt_items}
            adapt_df = pd.DataFrame(list(adapt_counts.items()), columns=["Adaptation Strategy", "Count"]).sort_values("Count", ascending=False)
            st.markdown("**Adaptation Strategies Against 10-Minute Delivery Apps:**")
            st.dataframe(adapt_df, use_container_width=True)

            ret_items = [
                "Easy return or exchange policies",
                "Free and quick home delivery",
                "Credit facility (Udhaar/Khata) for trusted customers",
                "Sourcing specific items on customer request",
                "Extended store operating hours"
            ]
            ret_counts = {r: df[COL_RETENTION_SERVICES].apply(lambda s: 1 if r in str(s) else 0).sum() for r in ret_items}
            ret_df = pd.DataFrame(list(ret_counts.items()), columns=["Retention Service", "Count"]).sort_values("Count", ascending=False)
            st.markdown("**Customer Retention Services Offered:**")
            st.dataframe(ret_df, use_container_width=True)

        st.markdown(
            """
            <div class="insight-card">
            <b>Summary:</b> <b>79.7% of stores</b> report sales drops from quick-delivery apps (37.5% moderate, 31.3% slight, 10.9% severe). 
            To retain customers, <b>51.6% offer direct WhatsApp/phone home delivery</b>, 37.5% provide easy return/exchange policies, and 28.1% offer store credit (Udhaar/Khata).
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 3. ID3 DECISION TREE
# =========================================================

elif nav_option.startswith("🌳 3.") or nav_option.startswith("3."):

    st.subheader("🌳 ID3 Decision Tree Classification")

    target_choice = st.selectbox(
        "Select Analytical Target Problem to Predict / Classify:",
        list(TARGET_OPTIONS.keys()),
        index=0,
        key="target_id3"
    )

    t_meta = TARGET_OPTIONS[target_choice]
    st.markdown(
        f"""
        <div class="insight-card" style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.12rem; font-weight: 800; color: #0F766E; margin-bottom: 0.45rem;">
                🎯 Business Challenge: {t_meta.get('title', target_choice)}
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.95rem;">
                <span style="color: #C2410C; font-weight: 700;">❓ Practical Question:</span> 
                <i style="color: #9A3412;">"{t_meta.get('business_question', t_meta['description'])}"</i>
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.92rem;">
                <span style="color: #B45309; font-weight: 700;">💼 Industry Application:</span> 
                <span style="color: #1E293B;">{t_meta.get('real_world_application', t_meta['description'])}</span>
            </div>
            <div style="font-size: 0.88rem;">
                <span style="color: #0E7490; font-weight: 700;">⚙️ Splitting Criterion:</span> 
                <span style="color: #115E59;">Shannon Information Gain (Maximizing Entropy Uncertainty Reduction)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Run ID3 Algorithm", key="btn_id3"):
        try:
            res = id3.run(target_name=target_choice)
            st.success("ID3 Model Executed Successfully.")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{res.get('accuracy', 0):.1f}%")
            with col2:
                st.metric("Criterion", "Entropy (Info Gain)")
            with col3:
                st.metric("Tree Depth", res.get("tree_depth", "N/A"))
            with col4:
                st.metric("Root Feature", str(res.get("root_attribute", "N/A")))

            st.markdown("### **Feature Importance (Information Gain)**")
            st.dataframe(res.get("feature_importance"), use_container_width=True)

            st.markdown("### **Decision Rules**")
            st.code(res.get("rules", ""), language="text")

            st.markdown("### **Test Set Predictions**")
            st.dataframe(res.get("data"), use_container_width=True)

            st.markdown("### **Analytical Takeaways**")
            for ins in res.get("business_insights", []):
                st.markdown(f"- {ins}")

        except Exception as e:
            st.error(f"ID3 Error: {e}")


# =========================================================
# 4. J48 DECISION TREE
# =========================================================

elif nav_option.startswith("🌿 4.") or nav_option.startswith("4."):

    st.subheader("🌿 J48 / C4.5 Decision Tree Classification")

    target_choice = st.selectbox(
        "Select Analytical Target Problem to Predict / Classify:",
        list(TARGET_OPTIONS.keys()),
        index=2,
        key="target_j48"
    )

    t_meta = TARGET_OPTIONS[target_choice]
    st.markdown(
        f"""
        <div class="insight-card" style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.12rem; font-weight: 800; color: #0F766E; margin-bottom: 0.45rem;">
                🎯 Business Challenge: {t_meta.get('title', target_choice)}
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.95rem;">
                <span style="color: #C2410C; font-weight: 700;">❓ Practical Question:</span> 
                <i style="color: #9A3412;">"{t_meta.get('business_question', t_meta['description'])}"</i>
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.92rem;">
                <span style="color: #B45309; font-weight: 700;">💼 Industry Application:</span> 
                <span style="color: #1E293B;">{t_meta.get('real_world_application', t_meta['description'])}</span>
            </div>
            <div style="font-size: 0.88rem;">
                <span style="color: #0E7490; font-weight: 700;">⚙️ Splitting Criterion:</span> 
                <span style="color: #115E59;">Shannon Entropy with Post-Pruned Leaf Regularization</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Run J48 Algorithm", key="btn_j48"):
        try:
            res = j48.run(target_name=target_choice)
            st.success("J48 Model Executed Successfully.")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{res.get('accuracy', 0):.1f}%")
            with col2:
                st.metric("Criterion", "Entropy (Pruned C4.5)")
            with col3:
                st.metric("Tree Depth", res.get("tree_depth", "N/A"))
            with col4:
                st.metric("Leaves", res.get("leaves", "N/A"))

            st.markdown("### **Feature Importance**")
            st.dataframe(res.get("importance"), use_container_width=True)

            st.markdown("### **Pruned Tree Structure**")
            st.code(res.get("rules", ""), language="text")

            st.markdown("### **Test Set Predictions**")
            st.dataframe(res.get("data"), use_container_width=True)

            st.markdown("### **Analytical Takeaways**")
            for ins in res.get("business_insights", []):
                st.markdown(f"- {ins}")

        except Exception as e:
            st.error(f"J48 Error: {e}")


# =========================================================
# 5. DECISION TREE (CART / GINI)
# =========================================================

elif nav_option.startswith("🌲 5.") or nav_option.startswith("5."):

    st.subheader("🌲 Decision Tree Classification (CART / Gini)")

    target_choice = st.selectbox(
        "Select Analytical Target Problem to Predict / Classify:",
        list(TARGET_OPTIONS.keys()),
        index=1,
        key="target_cart"
    )

    t_meta = TARGET_OPTIONS[target_choice]
    st.markdown(
        f"""
        <div class="insight-card" style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.12rem; font-weight: 800; color: #0F766E; margin-bottom: 0.45rem;">
                🎯 Business Challenge: {t_meta.get('title', target_choice)}
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.95rem;">
                <span style="color: #C2410C; font-weight: 700;">❓ Practical Question:</span> 
                <i style="color: #9A3412;">"{t_meta.get('business_question', t_meta['description'])}"</i>
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.92rem;">
                <span style="color: #B45309; font-weight: 700;">💼 Industry Application:</span> 
                <span style="color: #1E293B;">{t_meta.get('real_world_application', t_meta['description'])}</span>
            </div>
            <div style="font-size: 0.88rem;">
                <span style="color: #0E7490; font-weight: 700;">⚙️ Splitting Criterion:</span> 
                <span style="color: #115E59;">CART Gini Impurity Index & Gini Gain Reduction</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Run Decision Tree", key="btn_dt"):
        try:
            res = decision_tree.run(target_name=target_choice)
            st.success("Decision Tree Executed Successfully.")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{res.get('accuracy', 0):.1f}%")
            with col2:
                st.metric("Criterion", "Gini Impurity")
            with col3:
                st.metric("Tree Depth", res.get("tree_depth", "N/A"))
            with col4:
                st.metric("Root Feature", str(res.get("root_attribute", "N/A")))

            st.markdown("### **Gini Feature Importance**")
            st.dataframe(res.get("importance"), use_container_width=True)

            st.markdown("### **Decision Rules**")
            st.code(res.get("rules", ""), language="text")

            st.markdown("### **Test Set Predictions**")
            st.dataframe(res.get("data"), use_container_width=True)

            st.markdown("### **Analytical Takeaways**")
            for ins in res.get("business_insights", []):
                st.markdown(f"- {ins}")

        except Exception as e:
            st.error(f"Decision Tree Error: {e}")


# =========================================================
# 6. NAIVE BAYES CLASSIFICATION
# =========================================================

elif nav_option.startswith("🎯 6.") or nav_option.startswith("6."):

    st.subheader("🎯 Gaussian Naive Bayes Probabilistic Classification")

    target_choice = st.selectbox(
        "Select Analytical Target Problem to Predict / Classify:",
        list(TARGET_OPTIONS.keys()),
        index=4,
        key="target_nb"
    )

    t_meta = TARGET_OPTIONS[target_choice]
    st.markdown(
        f"""
        <div class="insight-card" style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.12rem; font-weight: 800; color: #0F766E; margin-bottom: 0.45rem;">
                🎯 Business Challenge: {t_meta.get('title', target_choice)}
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.95rem;">
                <span style="color: #C2410C; font-weight: 700;">❓ Practical Question:</span> 
                <i style="color: #9A3412;">"{t_meta.get('business_question', t_meta['description'])}"</i>
            </div>
            <div style="margin-bottom: 0.45rem; font-size: 0.92rem;">
                <span style="color: #B45309; font-weight: 700;">💼 Industry Application:</span> 
                <span style="color: #1E293B;">{t_meta.get('real_world_application', t_meta['description'])}</span>
            </div>
            <div style="font-size: 0.88rem;">
                <span style="color: #0E7490; font-weight: 700;">⚙️ Classification Rule:</span> 
                <span style="color: #115E59;">Bayes' Theorem with Maximum A Posteriori (MAP) Estimation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Run Naive Bayes", key="btn_nb"):
        try:
            res = naive_bayes.run(target_name=target_choice)
            st.success("Naive Bayes Executed Successfully.")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Accuracy", f"{res.get('accuracy', 0):.1f}%")
            with col2:
                priors = res.get("priors", {})
                prior_str = " | ".join([f"P({k}) = {v}%" for k, v in priors.items()])
                st.metric("Class Prior Probabilities", prior_str)

            st.markdown("### **Posterior Class Probabilities (Test Samples)**")
            st.dataframe(res.get("data"), use_container_width=True)

            st.markdown("### **Analytical Takeaways**")
            for ins in res.get("business_insights", []):
                st.markdown(f"- {ins}")

        except Exception as e:
            st.error(f"Naive Bayes Error: {e}")



# =========================================================
# 7. K-MEANS CLUSTERING
# =========================================================

elif nav_option.startswith("📍 7.") or nav_option.startswith("7."):

    st.subheader("📍 K-Means Operational Store Clustering")

    st.markdown(
        """
        - **Method:** K-Means ($k=3$) on standardized survey operational variables.
        - **Clustered Dimensions:** Demand impact on profit (1-5), pricing difficulty (1-5), supplier delay frequency (0-3), stock spoilage frequency (0-4), quick commerce impact (0-3), digital POS adoption (0/1), and restocking lead time (1-4).
        """
    )

    if st.button("Run K-Means Clustering", key="btn_kmeans"):
        try:
            res = kmeans.run()
            st.success("K-Means Clustering Completed Successfully.")

            st.markdown("### **Identified Store Profiles**")
            st.dataframe(res.get("cluster_counts"), use_container_width=True)

            st.markdown("### **Cluster Feature Averages**")
            st.dataframe(res.get("summary"), use_container_width=True)

            st.markdown("### **Store Cluster Assignments**")
            st.dataframe(res.get("data"), use_container_width=True)

            st.markdown("### **Operational Cluster Summaries**")
            for ins in res.get("business_insights", []):
                st.markdown(f"- {ins}")

        except Exception as e:
            st.error(f"K-Means Error: {e}")


# =========================================================
# 8. APRIORI ASSOCIATION RULES
# =========================================================

elif nav_option.startswith("🛒 8.") or nav_option.startswith("8."):

    st.subheader("🛒 Apriori Association Rule Mining")

    st.markdown(
        """
        - **Method:** Frequent itemset extraction across survey operational responses.
        - **Metrics:** Support (frequency of occurrence), Confidence (conditional probability), Lift (association strength beyond random chance).
        """
    )

    col_sup, col_conf = st.columns(2)
    with col_sup:
        min_sup = st.slider("Minimum Support Threshold (%)", min_value=15, max_value=40, value=25, step=5) / 100.0
    with col_conf:
        min_conf = st.slider("Minimum Confidence Threshold (%)", min_value=40, max_value=85, value=50, step=5) / 100.0

    if st.button("Run Apriori Mining", key="btn_apriori"):
        try:
            res_df = apriori.run(min_support=min_sup, min_confidence=min_conf, min_lift=1.05)

            if res_df.empty:
                st.warning("No rules found with these thresholds. Try lowering the support or confidence slider.")
            else:
                st.success(f"Discovered {len(res_df)} Association Rules.")

                st.markdown("### **Association Rules Table**")
                st.dataframe(res_df, use_container_width=True)

                st.markdown("### **Key Practical Patterns**")
                st.markdown(
                    """
                    1. **Substitute Products Driven by Supplier Delays:** Stores offering substitute products co-occur with frequent supplier delays in >90% of cases (Lift > 1.2), showing substitution is an emergency coping mechanism.
                    2. **WhatsApp Ordering Against Demand Swings:** Stores taking WhatsApp orders experience unpredictable demand in ~75% of cases (Lift ~1.3) to maintain direct contact with local customers.
                    3. **Safety Stock Buffer as Core Defensive Shield:** Maintaining buffer inventory co-occurs with unpredictable customer demand in ~73% of cases (Lift ~1.27).
                    """
                )

        except Exception as e:
            st.error(f"Apriori Error: {e}")



