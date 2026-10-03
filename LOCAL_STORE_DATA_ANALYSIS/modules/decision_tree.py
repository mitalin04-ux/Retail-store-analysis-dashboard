import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from modules.data_loader import get_feature_matrix, COL_STOCKOUT_LOSS, LABEL_STOCKOUT_LOSS


def run(target_name="Digital POS Adoption (Technology Modernization)"):
    """
    Standard CART Decision Tree using Gini Impurity on selected analytical target.
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
        criterion="gini",
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred) * 100)

    root_index = model.tree_.feature[0]
    root_attribute = X.columns[root_index] if root_index >= 0 else "None"

    rules = export_text(model, feature_names=list(X.columns))

    output = pd.DataFrame({
        f"Actual {target_name}": y_test.values,
        f"Predicted {target_name}": y_pred
    }, index=X_test.index)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Gini Importance": model.feature_importances_
    })
    importance["Gini Importance"] = importance["Gini Importance"].round(4)
    importance = importance[importance["Gini Importance"] > 0].sort_values(
        by="Gini Importance",
        ascending=False
    ).reset_index(drop=True)

    top_feature_str = f"**`{root_attribute}`**" if root_attribute != "None" else "operational variables"
    business_insights = [
        f"**Gini Root Split:** The model chose {top_feature_str} as the initial binary split that minimizes impurity for '{target_name}'.",
        "**Multi-Factor Interplay:** CART decision paths highlight how combinations of operational variables compound to drive store performance.",
        "**Actionable Insights:** Clear if-then branching provides transparent diagnostic rules for store owners."
    ]

    return {
        "data": output,
        "accuracy": accuracy,
        "criterion": "Gini Impurity (CART)",
        "tree_depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "root_attribute": root_attribute,
        "rules": rules,
        "target": target_name,
        "importance": importance,
        "feature_importance": importance,
        "business_insights": business_insights,
        "total_samples": len(raw_df),
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }