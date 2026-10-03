import pandas as pd
import numpy as np

CSV_PATH = "data/store_data.csv"

# Exact raw survey column mappings (grounded in CSV headers)
COL_TIMESTAMP = "Timestamp"
COL_EMAIL = "Email address"
COL_STORE_TYPE = "What type of store do you operate?"
COL_EXPERIENCE = "How long have you been running this store?"
COL_PROBLEMS = "What are the main problems you face when predicting sales and managing stock? (Select all that apply)"
COL_STOCK_ACTIONS = "What do you do when you have too much or too little stock?  (Select all that apply)"
COL_STOCKOUT_LOSS = "Have you ever lost sales because a product was unexpectedly out of stock?"
COL_DIGITAL_POS = "Do you use a digital system, such as a POS or inventory app, to keep track of your past sales?"
COL_DEMAND_IMPACT = "How much do unexpected changes in customer demand affect your shop's profits?"
COL_STOCK_WASTE = "How often do you have too much or too little stock that leads to products being wasted or spoiled?"
COL_DEMAND_HELP = "In your opinion, what would help you most in predicting customer demand?"
COL_QUICK_DELIVERY_IMPACT = "How much have quick delivery apps (e.g., Blinkit, Zepto, Swiggy Instamart) affected your shop's daily customers and sales?"
COL_ADAPTATION = "What have you done to compete with or adapt to 10-minute delivery apps? (Select all that apply)"
COL_PRICING_DIFFICULTY = "How difficult is it to maintain competitive pricing compared to larger supermarkets and online platforms?"
COL_PRICING_STRATEGY = "How do you usually change your prices to stay competitive while still making a profit?"
COL_SUPPLIER_DELAYS = "How often do your primary suppliers fail to deliver essential stock on time?"
COL_SUPPLIER_BACKUP = "When a key supplier faces a shortage, what is your immediate backup plan? (Select all that apply)"
COL_RETENTION_SERVICES = "Which additional services do you offer to retain customers who might otherwise shop online? (Select all that apply)"
COL_SOCIAL_MEDIA_SPIKES = "Have you experienced sudden spikes in demand for specific products due to social media trends (e.g., viral snacks, local trends, fashion trends)?"
COL_RESTOCK_SPEED = "If customers suddenly start asking for a new/trending product, how quickly can you usually get it in stock?"
COL_ADVICE = "What advice or strategy would you give to another local store owner struggling with overstocking, stockouts, or online competition?"

# Assertive descriptions for analysis and display
LABEL_STOCKOUT_LOSS = "Lost sales because a product was unexpectedly out of stock"
LABEL_DIGITAL_POS = "Digital system usage (POS / Inventory app) for sales tracking"
LABEL_DEMAND_IMPACT = "Impact of unexpected customer demand changes on shop profits"
LABEL_SUPPLIER_DELAYS = "Frequency of primary supplier delivery delays for essential stock"
LABEL_STOCK_WASTE = "Frequency of stock imbalance leading to product waste or spoilage"
LABEL_QUICK_DELIVERY = "Impact of quick delivery apps on daily customers and sales"
LABEL_PRICING_DIFFICULTY = "Difficulty of maintaining competitive pricing against supermarkets and online platforms"
LABEL_RESTOCK_SPEED = "Restocking speed for newly trending products"
LABEL_SOCIAL_MEDIA = "Experience of sudden demand spikes due to social media trends"


def load_clean_data():
    """
    Loads raw store_data.csv safely without modifying the CSV file.
    Normalizes column names, cleans text encoding artifacts, and handles whitespace.
    """
    try:
        df = pd.read_csv(CSV_PATH, encoding_errors="replace")
    except Exception:
        df = pd.read_csv(CSV_PATH, encoding="latin1")

    # Strip column names
    df.columns = df.columns.str.strip()

    # Clean text columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\ufffd", "-")
                .str.strip()
            )

    return df


# Dictionary of Practical Real-Life Business Target Problems
TARGET_OPTIONS = {
    "Stockout Sales Loss (Inventory Risk)": {
        "col": COL_STOCKOUT_LOSS,
        "title": "Inventory Stockout & Revenue Leakage Diagnostic",
        "description": "Predicts whether a store suffers direct revenue loss due to unexpected stockouts (Yes / No).",
        "business_question": "What operational factors (supplier delays, emergency reorders, manual auditing) cause local stores to lose paying customers to stockouts?",
        "real_world_application": "Enables retail store owners to identify high-risk product lines and implement buffer safety stocks before stockouts cause customer churn.",
        "type": "binary"
    },
    "Digital POS Adoption (Technology Modernization)": {
        "col": COL_DIGITAL_POS,
        "title": "Retail Digitization & POS Adoption Readiness",
        "description": "Predicts whether a store adopts modern digital POS / inventory software (Yes / No).",
        "business_question": "What store characteristics (operating age, pricing difficulty, quick delivery exposure) drive retailers to modernize from manual notebooks to digital POS apps?",
        "real_world_application": "Assists FinTech and SaaS providers (Khatabook, Dukaan, Square) in identifying mom-and-pop stores most primed for digital billing tools.",
        "type": "binary"
    },
    "Quick Commerce Threat Level (Market Disruption)": {
        "col": COL_QUICK_DELIVERY_IMPACT,
        "title": "Quick-Commerce Disruption & Footfall Vulnerability",
        "description": "Classifies stores suffering noticeable/severe sales drops from 10-minute delivery apps (Blinkit/Zepto) vs resilient stores.",
        "business_question": "Which local store categories are most vulnerable to losing daily footfall to 10-minute quick-delivery platforms?",
        "real_world_application": "Guides local merchants on whether they need to launch WhatsApp delivery, offer loyalty credit (Khata), or focus on perishable fresh goods to survive.",
        "type": "binary_qc"
    },
    "Profit Vulnerability to Demand Swings (Financial Risk)": {
        "col": COL_DEMAND_IMPACT,
        "title": "Profit Vulnerability to Demand Volatility",
        "description": "Classifies whether unforeseen demand swings heavily erode shop profit margins (High Risk 4-5 vs Manageable 1-3).",
        "business_question": "Why do certain local retailers experience heavy profit erosion during demand fluctuations while others remain financially resilient?",
        "real_world_application": "Assists store managers in stress-testing pricing strategies and markdown cycles to insulate net margins against viral trend fluctuations.",
        "type": "binary_demand"
    },
    "Frequent Supplier Delivery Delays (Supply Chain Fragility)": {
        "col": COL_SUPPLIER_DELAYS,
        "title": "Supply Chain Fragility & Supplier Failure Risk",
        "description": "Classifies stores suffering recurring supplier delivery failures (Weekly/Daily vs Reliable / Monthly).",
        "business_question": "Which store types and purchasing behaviors suffer chronic supplier delivery delays, and how effective are emergency backup plans?",
        "real_world_application": "Informs retailers when to establish secondary wholesale partnerships or group-buying syndicates to mitigate chronic delivery delays.",
        "type": "binary_supplier"
    }
}


def get_feature_matrix(target_key="Stockout Sales Loss (Inventory Risk)"):
    """
    Constructs a clean, interpretable operational feature matrix for classification.
    Prevents target leakage by excluding any feature representing the target variable.
    """
    df = load_clean_data()
    
    # Resolve target definition
    if target_key in TARGET_OPTIONS:
        t_meta = TARGET_OPTIONS[target_key]
        raw_col = t_meta["col"]
        t_type = t_meta["type"]
    else:
        raw_col = target_key
        t_type = "binary"

    # Prepare Target Series y
    if t_type == "binary":
        y = df[raw_col].astype(str).str.strip()
    elif t_type == "binary_qc":
        y = df[raw_col].apply(
            lambda x: "High Impact (Sales Drop)" if any(k in str(x) for k in ["Moderately", "Severely"]) else "Low/No Impact"
        )
    elif t_type == "binary_demand":
        y = pd.to_numeric(df[raw_col], errors="coerce").fillna(3).apply(
            lambda v: "High Profit Risk (4-5)" if v >= 4 else "Moderate/Low (1-3)"
        )
    elif t_type == "binary_supplier":
        y = df[raw_col].apply(
            lambda x: "Frequent Delays (Weekly/Daily)" if str(x).strip() in ["Weekly", "Daily"] else "Reliable / Monthly"
        )
    else:
        y = df[raw_col].astype(str).str.strip()

    X = pd.DataFrame(index=df.index)
    
    # Store Experience
    exp_map = {
        "Less than 1 year": 1,
        "1-3 years": 2,
        "4-7 years": 3,
        "8-10 years": 4,
        "More than 10 years": 5
    }
    X["Store_Experience_Level"] = df[COL_EXPERIENCE].map(exp_map).fillna(2).astype(int)
    
    # Digital POS
    if raw_col != COL_DIGITAL_POS:
        X["Digital_POS_Adoption"] = df[COL_DIGITAL_POS].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    
    # Demand Impact
    if raw_col != COL_DEMAND_IMPACT:
        X["Demand_Profit_Impact_1to5"] = pd.to_numeric(df[COL_DEMAND_IMPACT], errors="coerce").fillna(3).astype(int)
    
    # Stock Waste
    if raw_col != COL_STOCK_WASTE:
        waste_map = {"Rarely / Never": 0, "Seasonally": 1, "Monthly": 2, "Weekly": 3, "Daily": 4}
        X["Stock_Waste_Frequency"] = df[COL_STOCK_WASTE].map(waste_map).fillna(2).astype(int)
    
    # Quick Commerce Threat
    if raw_col != COL_QUICK_DELIVERY_IMPACT:
        qc_map = {
            "No impact (Sales remain steady)": 0,
            "Positively impacted (Partnered with platforms/increased sales)": 0,
            "Slightly impacted (Minor changes in customer behavior)": 1,
            "Moderately impacted (Noticeable but manageable drop)": 2,
            "Severely impacted (Significant drop in sales/footfall)": 3
        }
        X["Quick_Commerce_Threat_Score"] = df[COL_QUICK_DELIVERY_IMPACT].map(qc_map).fillna(1).astype(int)
    
    # Pricing Difficulty
    if raw_col != COL_PRICING_DIFFICULTY:
        X["Pricing_Difficulty_1to5"] = pd.to_numeric(df[COL_PRICING_DIFFICULTY], errors="coerce").fillna(3).astype(int)
    
    # Supplier Delays
    if raw_col != COL_SUPPLIER_DELAYS:
        delay_map = {"Rarely / Never": 0, "Monthly": 1, "Weekly": 2, "Daily": 3}
        X["Supplier_Delay_Frequency"] = df[COL_SUPPLIER_DELAYS].map(delay_map).fillna(1).astype(int)
    
    # Stockout Loss Feature (if not the target)
    if raw_col != COL_STOCKOUT_LOSS:
        X["Lost_Sales_From_Stockouts"] = df[COL_STOCKOUT_LOSS].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Social Media Demand Spikes
    if raw_col != COL_SOCIAL_MEDIA_SPIKES:
        X["Social_Media_Demand_Spikes"] = df[COL_SOCIAL_MEDIA_SPIKES].map({"Yes": 1, "No": 0}).fillna(1).astype(int)
    
    # Restocking Lead Time
    if raw_col != COL_RESTOCK_SPEED:
        restock_map = {
            "Within 1-2 days": 1,
            "Within a week": 2,
            "Within a month": 3,
            "I wait to see if the trend lasts before stocking": 4,
            "I generally do not stock short-term trend-based items": 5
        }
        X["Restocking_Lead_Time"] = df[COL_RESTOCK_SPEED].map(restock_map).fillna(2).astype(int)
    
    # Operational Multi-select Problems
    X["Problem_Unpredictable_Demand"] = df[COL_PROBLEMS].apply(lambda s: 1 if "Unpredictable changes" in str(s) else 0)
    X["Problem_Supplier_Delays"] = df[COL_PROBLEMS].apply(lambda s: 1 if "Supplier delays" in str(s) else 0)
    X["Problem_Seasonal_Fluctuations"] = df[COL_PROBLEMS].apply(lambda s: 1 if "Seasonal fluctuations" in str(s) else 0)
    X["Problem_Lack_Historical_Data"] = df[COL_PROBLEMS].apply(lambda s: 1 if "Lack of reliable historical" in str(s) else 0)
    X["Problem_Inadequate_Software"] = df[COL_PROBLEMS].apply(lambda s: 1 if "Inadequate software" in str(s) else 0)
    
    # Stock Management Actions
    X["Action_Safety_Stock"] = df[COL_STOCK_ACTIONS].apply(lambda s: 1 if "Maintain safety stock" in str(s) else 0)
    X["Action_Emergency_Orders"] = df[COL_STOCK_ACTIONS].apply(lambda s: 1 if "Place emergency orders" in str(s) else 0)
    X["Action_Discount_Promotions"] = df[COL_STOCK_ACTIONS].apply(lambda s: 1 if "Run discount promotions" in str(s) else 0)
    X["Action_Substitute_Products"] = df[COL_STOCK_ACTIONS].apply(lambda s: 1 if "Offer substitute products" in str(s) else 0)
    X["Action_Manual_Stock_Audits"] = df[COL_STOCK_ACTIONS].apply(lambda s: 1 if "Manually audit stock" in str(s) else 0)
    
    # Supplier Shortage Backup Actions
    X["Backup_Inform_Customers_Wait"] = df[COL_SUPPLIER_BACKUP].apply(lambda s: 1 if "Inform customers to wait" in str(s) else 0)
    X["Backup_Alternative_Brands"] = df[COL_SUPPLIER_BACKUP].apply(lambda s: 1 if "Switch to alternative brands" in str(s) else 0)
    X["Backup_Local_Wholesale_High_Cost"] = df[COL_SUPPLIER_BACKUP].apply(lambda s: 1 if "Buy from local wholesalers" in str(s) else 0)
    X["Backup_Borrow_Neighbors"] = df[COL_SUPPLIER_BACKUP].apply(lambda s: 1 if "Borrow stock" in str(s) else 0)

    # Store Type One-Hot Encoding
    store_types = pd.get_dummies(df[COL_STORE_TYPE], prefix="Type", dtype=int)
    X = pd.concat([X, store_types], axis=1)

    return df, X, y

