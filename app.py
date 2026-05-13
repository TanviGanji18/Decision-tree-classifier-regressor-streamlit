import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Title
st.title("🌳 Decision Tree - Classifier & Regressor")

# Mode selection
mode = st.sidebar.selectbox("Select Mode", ["Classification", "Regression"])

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Sidebar inputs
st.sidebar.header("🔧 Input Features")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    float(X.min().iloc[0]),
    float(X.max().iloc[0])
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    float(X.min().iloc[1]),
    float(X.max().iloc[1])
)

petal_length = st.sidebar.slider(
    "Petal Length",
    float(X.min().iloc[2]),
    float(X.max().iloc[2])
)

petal_width = st.sidebar.slider(
    "Petal Width",
    float(X.min().iloc[3]),
    float(X.max().iloc[3])
)

input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# =========================
# 📊 CLASSIFICATION MODE
# =========================
if mode == "Classification":

    st.subheader("📊 Classification Mode")

    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        'max_depth': [2, 3, 4, 5],
        'criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=params,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    prediction = model.predict(input_data)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"**Accuracy:** {acc:.4f}")
    st.write("**Predicted Class:**", iris.target_names[prediction][0])
    st.write("**Best Params:**", grid.best_params_)

# =========================
# 📈 REGRESSION MODE
# =========================
else:

    st.subheader("📈 Regression Mode")

    y = iris.data[:, 2]  # petal length

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10]
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid=params,
        cv=5,
        scoring='r2'
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    prediction = model.predict(input_data)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MSE:** {mse:.5f}")
    st.write(f"**R2 Score:** {r2:.5f}")
    st.write(f"**Predicted Petal Length:** {prediction[0]:.2f}")
    st.write("**Best Params:**", grid.best_params_)
