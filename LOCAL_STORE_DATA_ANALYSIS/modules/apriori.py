import pandas as pd
from itertools import combinations
from modules.data_loader import (
    load_clean_data,
    COL_STOCKOUT_LOSS,
    COL_DIGITAL_POS,
    COL_SOCIAL_MEDIA_SPIKES,
    COL_SUPPLIER_DELAYS,
    COL_QUICK_DELIVERY_IMPACT,
    COL_PROBLEMS,
    COL_STOCK_ACTIONS,
    COL_ADAPTATION,
    COL_SUPPLIER_BACKUP,
    COL_RETENTION_SERVICES
)


def run(min_support=0.20, min_confidence=0.50, min_lift=1.05):
    """
    Apriori Association Rule Mining on store survey transactions.
    Finds meaningful co-occurring operational practices, problems, and outcomes.
    """
    raw_df = load_clean_data()
    transactions = []

    for _, row in raw_df.iterrows():
        t = set()

        if str(row[COL_STOCKOUT_LOSS]).strip() == "Yes":
            t.add("Stockout Loss: Yes")

        if str(row[COL_DIGITAL_POS]).strip() == "Yes":
            t.add("Digital POS: Yes")
        else:
            t.add("Digital POS: No")

        if "Yes" in str(row[COL_SOCIAL_MEDIA_SPIKES]):
            t.add("Social Media Demand Spikes: Yes")

        delay_val = str(row[COL_SUPPLIER_DELAYS]).strip()
        if delay_val in ["Weekly", "Daily", "Monthly"]:
            t.add("Frequent Supplier Delays")

        qc_val = str(row[COL_QUICK_DELIVERY_IMPACT]).strip()
        if "Moderately" in qc_val or "Severely" in qc_val:
            t.add("High Quick Commerce Impact")

        problems_text = str(row[COL_PROBLEMS])
        for prob in [
            "Unpredictable changes in customer demand",
            "Supplier delays or inconsistent delivery",
            "Seasonal fluctuations and unexpected weather",
            "Lack of reliable historical sales data",
            "Inadequate software or inventory tracking tools"
        ]:
            if prob in problems_text:
                t.add(f"Problem: {prob}")

        actions_text = str(row[COL_STOCK_ACTIONS])
        for act in [
            "Maintain safety stock / buffer inventory",
            "Place emergency orders at higher supplier costs",
            "Run discount promotions to clear excess stock",
            "Offer substitute products to customers",
            "Manually audit stock levels more frequently"
        ]:
            if act in actions_text:
                t.add(f"Action: {act}")

        adapt_text = str(row[COL_ADAPTATION])
        for adapt in [
            "Offering home delivery via phone/WhatsApp orders",
            "Adopting digital payments (UPI) and modern billing",
            "Partnering directly with quick commerce or delivery platforms",
            "Enhancing personalized customer relationships and in-store experience"
        ]:
            if adapt in adapt_text:
                t.add(f"Adaptation: {adapt}")

        backup_text = str(row[COL_SUPPLIER_BACKUP])
        for bup in [
            "Inform customers to wait for the restock",
            "Switch to alternative brands or substitute products",
            "Buy from local wholesalers at a higher price",
            "Borrow stock from neighboring stores",
            "Source from a different city or region"
        ]:
            if bup in backup_text:
                t.add(f"Backup: {bup}")

        serv_text = str(row[COL_RETENTION_SERVICES])
        for serv in [
            "Easy return or exchange policies",
            "Free and quick home delivery",
            "Credit facility (Udhaar/Khata) for trusted customers",
            "Sourcing specific items on customer request",
            "Extended store operating hours"
        ]:
            if serv in serv_text:
                t.add(f"Service: {serv}")

        transactions.append(t)

    total_transactions = len(transactions)
    if total_transactions == 0:
        return pd.DataFrame()

    all_items = set().union(*transactions)
    rules_list = []

    for a, b in combinations(all_items, 2):
        both_count = sum({a, b}.issubset(t) for t in transactions)
        support = both_count / total_transactions

        if support < min_support:
            continue

        a_count = sum(a in t for t in transactions)
        b_count = sum(b in t for t in transactions)

        if a_count > 0:
            conf_a = both_count / a_count
            lift_a = conf_a / (b_count / total_transactions) if b_count > 0 else 0
            if conf_a >= min_confidence and lift_a >= min_lift:
                rules_list.append({
                    "If": a,
                    "Then": b,
                    "Support (%)": round(support * 100, 1),
                    "Confidence (%)": round(conf_a * 100, 1),
                    "Lift": round(lift_a, 2),
                    "Meaning": generate_rule_interpretation(a, b, conf_a, lift_a)
                })

        if b_count > 0:
            conf_b = both_count / b_count
            lift_b = conf_b / (a_count / total_transactions) if a_count > 0 else 0
            if conf_b >= min_confidence and lift_b >= min_lift:
                rules_list.append({
                    "If": b,
                    "Then": a,
                    "Support (%)": round(support * 100, 1),
                    "Confidence (%)": round(conf_b * 100, 1),
                    "Lift": round(lift_b, 2),
                    "Meaning": generate_rule_interpretation(b, a, conf_b, lift_b)
                })

    rules_df = pd.DataFrame(rules_list)
    if not rules_df.empty:
        rules_df = (
            rules_df.sort_values(by=["Lift", "Confidence (%)"], ascending=False)
            .drop_duplicates(subset=["If", "Then"])
            .reset_index(drop=True)
        )

    return rules_df


def generate_rule_interpretation(a, b, conf, lift):
    conf_pct = round(conf * 100, 1)
    if "Substitute products" in a and "Supplier Delays" in b:
        return f"Stores offering substitute products face frequent supplier delays in {conf_pct}% of cases (Lift {lift:.2f}), using substitution as an emergency backup."
    elif "WhatsApp orders" in a and "Unpredictable" in b:
        return f"Stores taking WhatsApp orders experience unpredictable demand in {conf_pct}% of cases (Lift {lift:.2f}), using direct chat to retain customers."
    elif "safety stock" in a.lower() and "demand" in b.lower():
        return f"Stores with safety stock report unpredictable demand in {conf_pct}% of cases (Lift {lift:.2f}), keeping buffers against unexpected surges."
    elif "Quick Commerce" in a and "WhatsApp" in b:
        return f"Stores disrupted by quick commerce adopt WhatsApp delivery in {conf_pct}% of cases (Lift {lift:.2f}) to compete locally."
    else:
        return f"When '{a}' occurs, '{b}' also occurs in {conf_pct}% of stores ({lift:.2f}x higher than random chance)."