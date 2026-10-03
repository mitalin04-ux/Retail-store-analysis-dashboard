import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from modules.data_loader import get_feature_matrix, COL_STOCKOUT_LOSS, LABEL_STOCKOUT_LOSS


def run(target_name="Quick Commerce Threat Level (Market Disruption)"):
    """
    J48 / C4.5 Decision Tree Classification with leaf size constraints.
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
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred) * 100)

    output = pd.DataFrame({
        f"Actual {target_name}": y_test.values,
        f"Predicted {target_name}": y_pred
    }, index=X_test.index)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Information Gain Score": model.feature_importances_
    })
    importance["Information Gain Score"] = importance["Information Gain Score"].round(4)
    importance = importance[importance["Information Gain Score"] > 0].sort_values(
        by="Information Gain Score",
        ascending=False
    ).reset_index(drop=True)

    root_index = model.tree_.feature[0]
    root_attribute = X.columns[root_index] if root_index >= 0 else "None"

    rules = export_text(model, feature_names=list(X.columns))

    top_feature_str = f"**`{root_attribute}`**" if root_attribute != "None" else "operational factors"
    business_insights = [
        f"**Pruned Tree Structure:** J48 generated a regularized {model.get_depth()}-level tree with {model.get_n_leaves()} leaves starting with {top_feature_str}.",
        f"**Target Analyzed:** '{target_name}' evaluated with `min_samples_leaf=2` to prevent overfitting on the survey dataset.",
        "**Practical Remedy:** Rule-based boundaries highlight actionable changes stores can take to improve operational resilience."
    ]

    return {
        "data": output,
        "accuracy": accuracy,
        "tree_depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "root_attribute": root_attribute,
        "importance": importance,
        "feature_importance": importance,
        "rules": rules,
        "criterion": "Entropy / C4.5-style (J48)",
        "target": target_name,
        "business_insights": business_insights,
        "total_samples": len(raw_df),
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }