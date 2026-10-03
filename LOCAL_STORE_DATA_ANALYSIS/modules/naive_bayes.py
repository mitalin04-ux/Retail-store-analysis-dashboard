import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from modules.data_loader import get_feature_matrix, COL_STOCKOUT_LOSS, LABEL_STOCKOUT_LOSS


def run(target_name="Frequent Supplier Delivery Delays (Supply Chain Fragility)"):
    """
    Gaussian Naive Bayes Probabilistic Classification on selected analytical target.
    """
    raw_df, X, y = get_feature_matrix(target_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(y.value_counts()) > 1 and y.value_counts().min() > 1 else None
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    accuracy = float(accuracy_score(y_test, y_pred) * 100)

    output = pd.DataFrame({
        f"Actual {target_name}": y_test.values,
        f"Predicted {target_name}": y_pred
    }, index=X_test.index)

    for i, class_name in enumerate(model.classes_):
        output[f"Probability {class_name} (%)"] = (y_prob[:, i] * 100).round(1)

    priors_dict = {
        str(cls): round(prior * 100, 1)
        for cls, prior in zip(model.classes_, model.class_prior_)
    }

    priors_summary = ", ".join([f"**P({k}) = {v}%**" for k, v in priors_dict.items()])
    business_insights = [
        f"**Dataset Prior Distribution:** Baseline class probabilities for '{target_name}': {priors_summary}.",
        "**Bayesian Conditional Risk:** Updates prior likelihood based on joint operational evidence (delay frequency, demand swings, pricing pressure).",
        "**Diagnostic Utility:** Identifies which operational risk signals tilt posterior probabilities most heavily toward vulnerable categories."
    ]

    return {
        "data": output,
        "accuracy": accuracy,
        "classes": list(model.classes_),
        "priors": priors_dict,
        "target": target_name,
        "algorithm": "Gaussian Naive Bayes",
        "business_insights": business_insights,
        "total_samples": len(raw_df),
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }