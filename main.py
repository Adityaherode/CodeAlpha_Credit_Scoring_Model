import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load dataset
    df = pd.read_csv("data/credit_data.csv")

    # Exploratory Data Analysis (EDA)
    print("📊 Dataset Preview:")
    print(df.head())
    print("\nClass balance:\n", df["default"].value_counts())

    sns.countplot(x="default", data=df)
    plt.title("Default Distribution")
    plt.savefig("data/default_distribution.png")

    # Feature Engineering
    df["debt_to_income_ratio"] = df["debt"] / df["income"]

    # Split features and target
    X = df.drop("default", axis=1)
    y = df["default"]

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    # Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n🔹 {name} Results:")
        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_pred))

if __name__ == "__main__":
    main()