
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "train.csv"

# Load the trained machine learning model
model = joblib.load(MODEL_PATH)

# Load the actual Titanic dataset
data = pd.read_csv(DATA_PATH)

# Calculate dashboard statistics from the dataset
total = len(data)
survived = int(data["Survived"].sum())
not_survived = total - survived

stats = {
    "total": total,
    "survived": survived,
    "not_survived": not_survived,
    "rate": round((survived / total) * 100, 1)
}

# Calculate survival rates by gender
gender_rates = (
    data.groupby("Sex")["Survived"].mean() * 100
).round(1)

gender_stats = {
    "female": float(gender_rates.get("female", 0)),
    "male": float(gender_rates.get("male", 0))
}

# Calculate survival rates by passenger class
class_rates = (
    data.groupby("Pclass")["Survived"].mean() * 100
).round(1)

class_stats = {
    "first": float(class_rates.get(1, 0)),
    "second": float(class_rates.get(2, 0)),
    "third": float(class_rates.get(3, 0))
}

# Chart data
gender_chart = {
    "labels": ["Female", "Male"],
    "values": [
        gender_stats["female"],
        gender_stats["male"]
    ]
}

class_chart = {
    "labels": ["1st Class", "2nd Class", "3rd Class"],
    "values": [
        class_stats["first"],
        class_stats["second"],
        class_stats["third"]
    ]
}


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:
            passenger = {
                "pclass": int(request.form["pclass"]),
                "age": float(request.form["age"]),
                "sex": request.form["sex"],
                "fare": float(request.form["fare"]),
                "sibsp": int(request.form["sibsp"]),
                "parch": int(request.form["parch"]),
                "embarked": request.form["embarked"]
            }

        except (KeyError, ValueError):
            error = "Please complete all fields with valid values."

        else:

            if (
                passenger["pclass"] not in (1, 2, 3)
                or not math.isfinite(passenger["age"])
                or not 0 <= passenger["age"] <= 120
                or passenger["sex"] not in ("female", "male")
                or not math.isfinite(passenger["fare"])
                or passenger["fare"] < 0
                or not 0 <= passenger["sibsp"] <= 20
                or not 0 <= passenger["parch"] <= 20
                or passenger["embarked"] not in ("C", "Q", "S")
            ):
                error = "Please check the passenger details and enter valid values."

            else:

                # Prepare input features for the trained model
                feature_values = {
                    "Pclass": passenger["pclass"],
                    "Age": passenger["age"],
                    "SibSp": passenger["sibsp"],
                    "Parch": passenger["parch"],
                    "Sex_female": int(passenger["sex"] == "female"),
                    "Sex_male": int(passenger["sex"] == "male"),
                    "Embarked_C": int(passenger["embarked"] == "C"),
                    "Embarked_Q": int(passenger["embarked"] == "Q"),
                    "Embarked_S": int(passenger["embarked"] == "S")
                }

                try:
                    # Preserve the exact feature order used during training
                    if hasattr(model, "feature_names_in_"):
                        feature_columns = list(model.feature_names_in_)
                    else:
                        feature_columns = list(feature_values.keys())

                    features = pd.DataFrame(
                        [feature_values],
                        columns=feature_columns
                    )

                    # Generate prediction
                    prediction = int(model.predict(features)[0])

                    # Calculate probability of survival
                    classes = list(model.classes_)
                    survival_index = classes.index(1)

                    probability = float(
                        model.predict_proba(features)[0][survival_index]
                    )

                    passenger["embarked_name"] = {
                        "C": "Cherbourg",
                        "Q": "Queenstown",
                        "S": "Southampton"
                    }[passenger["embarked"]]

                    result = {
                        "survived": prediction == 1,
                        "probability": probability,
                        "passenger": passenger
                    }

                except (ValueError, KeyError, AttributeError) as exc:
                    app.logger.exception("Prediction failed")
                    error = "Prediction could not be completed. Please check the model and its input features."

    return render_template(
        "index.html",
        stats=stats,
        gender_stats=gender_stats,
        class_stats=class_stats,
        gender_chart=gender_chart,
        class_chart=class_chart,
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
