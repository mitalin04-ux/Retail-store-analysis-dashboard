import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from modules.data_loader import get_feature_matrix, COL_STOCKOUT_LOSS, LABEL_STOCKOUT_LOSS


def run(target_name="Stockout Sales Loss (Inventory Risk)"):
    """
    ID3 Classification using Entropy / Information Gain on selected analytical target.
    """
    raw_df, X, y = get_feature_matrix(target_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(y.value_counts()) > 1 and y.value_counts().min() > 1 else None
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred) * 100)

    result_df = pd.DataFrame({
        f"Actual {target_name}": y_test.values,
        f"Predicted {target_name}": y_pred
    }, index=X_test.index)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Information Gain Score": model.feature_importances_
    })
    importance_df["Information Gain Score"] = importance_df["Information Gain Score"].round(4)
    importance_df = importance_df[importance_df["Information Gain Score"] > 0].sort_values(
        by="Information Gain Score",
        ascending=False
    ).reset_index(drop=True)

    root_index = model.tree_.feature[0]
    root_feature = X.columns[root_index] if root_index >= 0 else "None"

    rules_text = export_text(model, feature_names=list(X.columns))

    top_feature_str = f"**`{root_feature}`**" if root_feature != "None" else "operational variables"
    business_insights = [
        f"**Root Splitting Factor:** The ID3 algorithm selected {top_feature_str} as the attribute with the maximum Information Gain.",
        f"**Target Breakdown:** Classified {len(raw_df)} stores across '{target_name}' using pure entropy decision logic.",
        "**Practical Significance:** Operational characteristics successfully separate retail business outcomes without complex black-box models."
    ]

    return {
        "data": result_df,
        "accuracy": accuracy,
        "algorithm": "ID3 (Entropy Decision Tree)",
        "criterion": "Entropy / Information Gain",
        "tree_depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "root_attribute": root_feature,
        "feature_importance": importance_df,
        "importance": importance_df,
        "rules": rules_text,
        "target": target_name,
        "business_insights": business_insights,
        "total_samples": len(raw_df),
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }