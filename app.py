import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Analyzing Amazon EC2 Instance Costs")

file_path = "ec2dataset.csv"
data = pd.read_csv(file_path)


st.header("Dataset Information")

dataset_information = pd.DataFrame(
    {
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Non-Null Values": data.notnull().sum().values,
        "Missing Values": data.isnull().sum().values
    }
)

st.dataframe(
    dataset_information,
    use_container_width=True,
    hide_index=True
)



st.header("Dataset Preview")

st.dataframe(
    data.head(),
    use_container_width=True
)

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace(
            "[$, hourly]",
            "",
            regex=True
        ),
        errors="coerce"
    )



st.header("Missing Values After Conversion")

missing_values = (
    data[cost_columns]
    .isnull()
    .sum()
)

st.dataframe(
    missing_values.to_frame(
        name="Missing Values"
    ),
    use_container_width=True
)



st.header("Cost Summary Statistics")

cost_summary = data[cost_columns].describe()

st.dataframe(
    cost_summary,
    use_container_width=True
)


st.header("EC2 Cost Comparison")

sns.set(style="whitegrid")

fig1, ax1 = plt.subplots(
    figsize=(12, 6)
)

sns.boxplot(
    data=data[cost_columns],
    palette="Set2",
    ax=ax1
)

ax1.set_title(
    "Cost Comparison of Amazon EC2 Instances (Hourly)",
    fontsize=16
)

ax1.set_ylabel(
    "Cost (USD)",
    fontsize=12
)

ax1.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


st.header("On-Demand Cost Outliers")


def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]


outliers_on_demand = detect_outliers(
    "On Demand"
)

st.write(
    f"{len(outliers_on_demand)} On-Demand cost outliers found"
)

st.dataframe(
    outliers_on_demand,
    use_container_width=True
)


st.header("On-Demand and Reserved Cost Comparison")

cost_comparison = (
    data[
        [
            "Name",
            "On Demand",
            "Linux Reserved cost"
        ]
    ]
    .dropna()
    .sort_values("On Demand")
)

st.write("Ten Lowest-Cost Instances")

st.dataframe(
    cost_comparison.head(10),
    use_container_width=True
)


def filter_instance_family(family):
    return data[
        data["Name"].str.startswith(family)
    ]


t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")


st.header("T2 and T3 Instance Comparison")

st.subheader("T2 Instance Costs Summary")

t2_summary = (
    t2_instances[cost_columns]
    .describe()
)

st.dataframe(
    t2_summary,
    use_container_width=True
)


st.subheader("T3 Instance Costs Summary")

t3_summary = (
    t3_instances[cost_columns]
    .describe()
)

st.dataframe(
    t3_summary,
    use_container_width=True
)


st.header("T2 Instance Cost Distribution")

fig2, ax2 = plt.subplots(
    figsize=(12, 6)
)

sns.boxplot(
    data=t2_instances[cost_columns],
    palette="Blues",
    showmeans=True,
    ax=ax2
)

ax2.set_title(
    "Cost Distribution for T2 Instances",
    fontsize=16
)

ax2.set_ylabel(
    "Cost (USD)",
    fontsize=12
)

ax2.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(fig2)

plt.close(fig2)


st.header("T3 Instance Cost Distribution")

fig3, ax3 = plt.subplots(
    figsize=(12, 6)
)

sns.boxplot(
    data=t3_instances[cost_columns],
    palette="Greens",
    showmeans=True,
    ax=ax3
)

ax3.set_title(
    "Cost Distribution for T3 Instances",
    fontsize=16
)

ax3.set_ylabel(
    "Cost (USD)",
    fontsize=12
)

ax3.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(fig3)

plt.close(fig3)


st.header("Lowest-Cost T2 and T3 Instances")

comparison = pd.concat(
    [
        t2_instances[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ],
        t3_instances[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ]
    ]
)

comparison_sorted = (
    comparison
    .dropna()
    .sort_values("On Demand")
)

st.dataframe(
    comparison_sorted.head(10),
    use_container_width=True
)

#--------------Part 2-------------
st.header("Part 2: Predicting EC2 Instance Costs")

st.write(
    "This section uses EC2 memory and vCPU values to predict "
    "the hourly On-Demand cost."
)


st.subheader("Feature Engineering")

# Create a copy so Part 1 data is not changed
regression_data = data.copy()

# Convert memory from text into a numeric value
regression_data["Instance Memory"] = pd.to_numeric(
    regression_data["Instance Memory"]
    .str.replace(" GiB", "")
)

# Extract the number of vCPUs
regression_data["vCPUs"] = pd.to_numeric(
    regression_data["vCPUs"]
    .str.extract(
        r"(\d+)",
        expand=False
    )
)

st.write("Converted Memory and vCPU Values")

st.dataframe(
    regression_data[
        [
            "Instance Memory",
            "vCPUs"
        ]
    ].head(),
    use_container_width=True
)


st.subheader("Cleaned Regression Data")

data_cleaned = regression_data.dropna(
    subset=[
        "On Demand",
        "Instance Memory",
        "vCPUs"
    ]
)

missing_regression_values = (
    data_cleaned[
        [
            "On Demand",
            "Instance Memory",
            "vCPUs"
        ]
    ]
    .isnull()
    .sum()
)

st.dataframe(
    missing_regression_values.to_frame(
        name="Missing Values"
    ),
    use_container_width=True
)



st.subheader("Training and Testing Data")

X = data_cleaned[["Instance Memory", "vCPUs"]]

# Train using the logarithm of cost to prevent unrealistic negative predictions
y = np.log1p(data_cleaned["On Demand"])

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

training_column, testing_column = st.columns(2)

training_column.metric(
    "Training Samples",
    len(X_train)
)

testing_column.metric(
    "Testing Samples",
    len(X_test)
)


st.subheader("Linear Regression Model")

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

st.write(
    f"Intercept: {model.intercept_}"
)

coefficient_data = pd.DataFrame(
    {
        "Feature": [
            "Instance Memory",
            "vCPUs"
        ],
        "Coefficient": model.coef_
    }
)

st.write("Model Coefficients")

st.dataframe(
    coefficient_data,
    use_container_width=True,
    hide_index=True
)


st.subheader("Model Evaluation")

# Predict the costs for the testing data
# Convert predictions back to regular dollar amounts
y_pred = np.expm1(model.predict(X_test))
y_test_actual = np.expm1(y_test)

# Calculate performance metrics
mae = mean_absolute_error(
    y_test_actual,
    y_pred
)

mse = mean_squared_error(
    y_test_actual,
    y_pred
)

rmse = mse ** 0.5

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Mean Absolute Error",
    f"{mae:.4f}"
)

metric2.metric(
    "Mean Squared Error",
    f"{mse:.4f}"
)

metric3.metric(
    "Root Mean Squared Error",
    f"{rmse:.4f}"
)


st.subheader("Actual vs Predicted On-Demand Costs")

fig4, ax4 = plt.subplots(
    figsize=(8, 6)
)

ax4.scatter(
    y_test_actual,
    y_pred,
    alpha=0.7,
    color="blue"
)

minimum_cost = min(
    y_test_actual.min(),
    y_pred.min()
)

maximum_cost = max(
    y_test_actual.max(),
    y_pred.max()
)

ax4.plot(
    [minimum_cost, maximum_cost],
    [minimum_cost, maximum_cost],
    color="red",
    linestyle="--"
)

ax4.set_title(
    "Actual vs Predicted On-Demand Costs"
)

ax4.set_xlabel(
    "Actual On-Demand Cost"
)

ax4.set_ylabel(
    "Predicted On-Demand Cost"
)

plt.tight_layout()

st.pyplot(fig4)

plt.close(fig4)

st.subheader("Predict EC2 On-Demand Cost")

memory_input = st.number_input(
    "Instance Memory (GiB)",
    min_value=0.5,
    value=4.0,
    step=0.5
)

vcpu_input = st.number_input(
    "Number of vCPUs",
    min_value=1,
    value=2,
    step=1
)

if st.button("Predict Cost"):
    new_instance = pd.DataFrame(
        [[memory_input, vcpu_input]],
        columns=["Instance Memory", "vCPUs"]
    )

    # The model returns a logarithmic prediction
    log_prediction = model.predict(new_instance)[0]

    # Convert the logarithmic prediction back into dollars
    predicted_cost = np.expm1(log_prediction)

    # Additional protection against impossible negative prices
    predicted_cost = max(0.0, predicted_cost)

    st.success(
        f"Predicted On-Demand Cost: ${predicted_cost:.4f} per hour"
    )
