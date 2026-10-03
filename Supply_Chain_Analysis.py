import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# COLOUR THEME

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

viridis_colors = cm.viridis(np.linspace(0, 1, 5))

primary_color = viridis_colors[0]
secondary_color = viridis_colors[1]
accent_color = viridis_colors[3]
danger_color = "#F50C0C"
neutral_color = viridis_colors[4]

custom_palette = viridis_colors


# LOADING DATA

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1"
)


# BASIC INFORMATION

print("\nDataset loaded successfully!")

print("\nColumns:")
print(df.columns.tolist())

print("\nRows, Columns:")
print(df.shape)

print("\nNumber of duplicates:")
print(df.duplicated().sum())

print("\nMissing Values:")
print(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)


# DATA CLEANING

columns_to_drop = [
    'Product Description',
    'Product Image',
    'Customer Email',
    'Customer Password',
    'Customer Fname',
    'Customer Lname',
    'Customer Street',
    'Customer Zipcode',
    'Order Zipcode',
    'Longitude',
    'Latitude',
    'Order Item Cardprod Id',
    'Order Item Id',
    'Order Item Discount',
    'Order Item Discount Rate',
    'Order Item Product Price',
    'Order Item Total',
    'Category Id',
    'Department Id',
    'Order Id',
    'Order Customer Id',
    'Customer Id',
    'Product Card Id',
    'Product Category Id',
    'Benefit per order',
    'Product Status',
    'Customer City',
    'Order Country',
    'Order State',
    'Customer State',
    'Market'
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# REMOVE CANCELLED ORDERS

df = df[
    df['Delivery Status'] != 'Shipping canceled'
]


# STANDARD DATE CONVERSION

date_columns = [
    'order date (DateOrders)',
    'shipping date (DateOrders)'
]

for column in date_columns:

    df[column] = pd.to_datetime(
        df[column],
        errors='coerce'
    )


# REMOVE DUPLICATES

print(
    "\nDuplicates before removing:",
    df.duplicated().sum()
)

df = df.drop_duplicates()

print(
    "Duplicates after removing:",
    df.duplicated().sum()
)


# REMOVE INVALID DATES

df = df.dropna(
    subset=date_columns
)


# OVERVIEW AFTER CLEANING

print("\nRows, Columns:")
print(df.shape)

print("\nMissing Values:")
print(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nData Types:")
print(df.dtypes)

print("\nFinal Columns:")
print(df.columns.tolist())


# LOW CARDINALITY COLUMNS

for column in df.columns:

    if df[column].nunique() < 10:

        print(
            f"\n{column} value counts:"
        )

        print(
            df[column].value_counts()
        )


# ORDER PROCESSING TIME

df['Order Processing Time'] = (
    df['shipping date (DateOrders)']
    - df['order date (DateOrders)']
).dt.days


# DELIVERY DELAY

df['Delay'] = (
    df['Order Processing Time']
    - df['Days for shipment (scheduled)']
)


# DELAY FLAG

df['Is_Delayed'] = (
    df['Delay'] > 0
)


# TIME FEATURES

df['order_month'] = (
    df['order date (DateOrders)']
    .dt.month
)

df['order_day'] = (
    df['order date (DateOrders)']
    .dt.day_name()
)

df['order_hour'] = (
    df['order date (DateOrders)']
    .dt.hour
)


# BASIC STATISTICS

print("\nDescriptive Statistics:")

print(
    df.describe()
)


print("\nDelayed vs On-time Orders:")

print(
    df['Is_Delayed']
    .value_counts()
)


# PROFITABILITY FLAG

df["Profitability Flag"] = np.where(
    df['Order Profit Per Order'] > 0,
    'Profit',
    np.where(
        df['Order Profit Per Order'] < 0,
        'Loss',
        'Break-even'
    )
)


print(
    "\nProfitability Distribution:"
)

print(
    df['Profitability Flag']
    .value_counts()
)


# PROFITABILITY VISUALIZATION

profit_counts = (
    df['Profitability Flag']
    .value_counts(normalize=True)
    * 100
)


plt.figure(
    figsize=(6, 6)
)

profit_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=[
        accent_color,
        danger_color,
        secondary_color
    ]
)

plt.ylabel("")

plt.title(
    "Profitability Distribution (%)"
)

plt.tight_layout()

plt.show()


# FORMAT FUNCTION

def format_func(
    value,
    tick_number=None
):

    if abs(value) >= 1e6:

        return f'{value / 1e6:.1f}M $'

    elif abs(value) >= 1e3:

        return f'{value / 1e3:.1f}K $'

    else:

        return f'{value:.0f} $'


# BUSINESS KPIs

delayed_df = df[
    df['Delay'] > 0
]


metrics = {}

metrics['Total Orders'] = len(df)

metrics['Late Deliveries'] = len(
    delayed_df
)

metrics['90% Delay (days)'] = (
    delayed_df['Delay']
    .quantile(0.90)
)

metrics['On time Delivery %'] = (
    1
    - (
        len(delayed_df)
        / len(df)
    )
) * 100

metrics['Late Delivery %'] = (
    len(delayed_df)
    / len(df)
) * 100

metrics['Total Profit'] = format_func(
    df['Order Profit Per Order']
    .sum()
)

metrics['Profit from Delayed Orders'] = (
    format_func(
        df.loc[
            df['Delay'] > 0,
            'Order Profit Per Order'
        ].sum()
    )
)


print(
    "\n########## BUSINESS KPIs #########"
)

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.2f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# PROFITABILITY VS DELIVERY TIME

profit_metrics = (
    df.groupby('Delay')
    ['Order Profit Per Order']
    .agg(
        mean_profit='mean',
        total_profit='sum',
        order_count='count'
    )
    .reset_index()
)


delay_distribution = (
    df['Delay']
    .value_counts(normalize=True)
    .sort_index()
    * 100
).reset_index()


delay_distribution.columns = [
    'Delay_days',
    'Percentage'
]


print(
    "\nProfit Metrics by Delay:"
)

print(
    profit_metrics.round(2)
)


print(
    "\nDelay Distribution:"
)

print(
    delay_distribution.round(2)
)


# PROFITABILITY BY DELAY VISUALIZATION

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)


sns.barplot(
    x='Delay_days',
    y='Percentage',
    data=delay_distribution,
    color=accent_color,
    ax=ax1
)


ax1.set_title(
    "Delay Distribution"
)

ax1.set_xlabel(
    "Delay (days)"
)

ax1.set_ylabel(
    "Percentage of Orders (%)"
)


for bar in ax1.patches:

    height = bar.get_height()

    if height > 0:

        ax1.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            fontsize=9
        )


ax2.bar(
    profit_metrics['Delay'],
    profit_metrics['total_profit'],
    color=primary_color,
    alpha=0.8,
    label="Total Profit"
)


ax2.set_xlabel(
    "Delay Days"
)

ax2.set_ylabel(
    "Total Profit"
)

ax2.set_title(
    "Profitability vs Delay"
)

ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(format_func)
)


ax3 = ax2.twinx()


ax3.plot(
    profit_metrics['Delay'],
    profit_metrics['mean_profit'],
    color=accent_color,
    marker='o',
    linewidth=2,
    label="Mean Profit"
)


ax3.set_ylabel(
    "Mean Profit"
)


lines1, labels1 = (
    ax2.get_legend_handles_labels()
)

lines2, labels2 = (
    ax3.get_legend_handles_labels()
)


ax3.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right"
)


plt.tight_layout()

plt.show()


# BOTTLENECK ANALYSIS

def compute_delay_pct_by_category(
    category
):

    category_df = (
        df.groupby(category)
        .agg(
            total_orders=(
                'Delay',
                'count'
            ),
            late_orders=(
                'Is_Delayed',
                'sum'
            )
        )
        .reset_index()
    )

    category_df['delay_pct'] = (
        category_df['late_orders']
        / category_df['total_orders']
        * 100
    )

    return (
        category_df
        .sort_values(
            'delay_pct',
            ascending=False
        )
        .head(10)
    )


categories = [
    'Order Region',
    'Customer Segment',
    'Shipping Mode',
    'Order Status',
    'Type',
    'Department Name'
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 8),
    constrained_layout=True
)


axes = axes.flatten()


for ax, category in zip(
    axes,
    categories
):

    category_df = (
        compute_delay_pct_by_category(
            category
        )
    )

    sns.barplot(
        data=category_df,
        x='delay_pct',
        y=category,
        color=accent_color,
        ax=ax
    )

    ax.set_title(
        f"Delay % by {category}"
    )

    ax.set_xlabel(
        "Delay Percentage (%)"
    )

    ax.set_ylabel(
        category
    )


plt.show()


# BOTTLENECK TABLES

print(
    "\n########## BOTTLENECK ANALYSIS #########"
)

for category in categories:

    category_df = (
        compute_delay_pct_by_category(
            category
        )
    )

    print(
        f"\nDelay % by {category}:"
    )

    print(
        category_df
        .round(2)
        .to_string(index=False)
    )


# ROOT CAUSE ANALYSIS

def top_driver_for_region(
    region
):

    region_df = df[
        df['Order Region'] == region
    ].copy()

    drivers = [
        "Shipping Mode",
        "Customer Segment",
        "Department Name",
        "Type",
        "Order Status"
    ]

    all_factors = []


    for factor in drivers:

        temp = (
            region_df
            .groupby(factor)
            .agg(
                total_orders=(
                    'Delay',
                    'count'
                ),
                late_orders=(
                    'Is_Delayed',
                    'sum'
                ),
                avg_delay=(
                    'Delay',
                    'mean'
                )
            )
            .reset_index()
        )

        temp['delay_pct'] = (
            temp['late_orders']
            / temp['total_orders']
            * 100
        )

        temp['Driver'] = factor

        temp['Factor_level'] = (
            factor
            + ": "
            + temp[factor].astype(str)
        )

        all_factors.append(
            temp[
                [
                    'Driver',
                    'Factor_level',
                    'delay_pct',
                    'avg_delay',
                    'total_orders'
                ]
            ]
        )


    final_df = pd.concat(
        all_factors,
        ignore_index=True
    )


    top_factors = (
        final_df
        .sort_values(
            'delay_pct',
            ascending=False
        )
        .head(10)
    )


    plt.figure(
        figsize=(10, 6)
    )


    bars = plt.barh(
        top_factors['Factor_level'],
        top_factors['delay_pct'],
        color=accent_color
    )


    plt.xlabel(
        "Delay Percentage (%)"
    )

    plt.ylabel(
        "Driver"
    )

    plt.title(
        f"Delivery Delay Drivers - {region}"
    )

    plt.gca().invert_yaxis()


    for bar in bars:

        width = bar.get_width()

        plt.text(
            width - 5,
            bar.get_y()
            + bar.get_height() / 2,
            f"{width:.1f}%",
            va='center',
            color='white'
        )


    plt.tight_layout()

    plt.show()


top_driver_for_region(
    'Central Africa'
)

top_driver_for_region(
    'East Africa'
)


# TIME BASED ANALYSIS

delay_by_month = (
    df.groupby('order_month')
    ['Is_Delayed']
    .mean()
    .reset_index()
)

delay_by_month['delay_pct'] = (
    delay_by_month['Is_Delayed']
    * 100
)


delay_by_day = (
    df.groupby('order_day')
    ['Is_Delayed']
    .mean()
    .reset_index()
)

delay_by_day['delay_pct'] = (
    delay_by_day['Is_Delayed']
    * 100
)


delay_by_hour = (
    df.groupby('order_hour')
    ['Is_Delayed']
    .mean()
    .reset_index()
)

delay_by_hour['delay_pct'] = (
    delay_by_hour['Is_Delayed']
    * 100
)


day_order = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]


delay_by_day['order_day'] = pd.Categorical(
    delay_by_day['order_day'],
    categories=day_order,
    ordered=True
)


delay_by_day = (
    delay_by_day
    .sort_values('order_day')
)


# TIME BASED TABLES

print(
    "\n########## TIME BASED ANALYSIS #########"
)

print(
    "\nDelay % by Month:"
)

print(
    delay_by_month
    .round(2)
    .to_string(index=False)
)


print(
    "\nDelay % by Day:"
)

print(
    delay_by_day
    .round(2)
    .to_string(index=False)
)


print(
    "\nDelay % by Hour:"
)

print(
    delay_by_hour
    .round(2)
    .to_string(index=False)
)


# HIGHEST AND LOWEST PERIODS

highest_delay_month = (
    delay_by_month
    .loc[
        delay_by_month['delay_pct']
        .idxmax()
    ]
)

lowest_delay_month = (
    delay_by_month
    .loc[
        delay_by_month['delay_pct']
        .idxmin()
    ]
)


highest_delay_day = (
    delay_by_day
    .loc[
        delay_by_day['delay_pct']
        .idxmax()
    ]
)

lowest_delay_day = (
    delay_by_day
    .loc[
        delay_by_day['delay_pct']
        .idxmin()
    ]
)


highest_delay_hour = (
    delay_by_hour
    .loc[
        delay_by_hour['delay_pct']
        .idxmax()
    ]
)

lowest_delay_hour = (
    delay_by_hour
    .loc[
        delay_by_hour['delay_pct']
        .idxmin()
    ]
)


print(
    "\n########## HIGHEST DELAY PERIODS #########"
)

print(
    f"Highest Delay Month: "
    f"{int(highest_delay_month['order_month'])} "
    f"({highest_delay_month['delay_pct']:.2f}%)"
)

print(
    f"Highest Delay Day: "
    f"{highest_delay_day['order_day']} "
    f"({highest_delay_day['delay_pct']:.2f}%)"
)

print(
    f"Highest Delay Hour: "
    f"{int(highest_delay_hour['order_hour'])}:00 "
    f"({highest_delay_hour['delay_pct']:.2f}%)"
)


print(
    "\n########## LOWEST DELAY PERIODS #########"
)

print(
    f"Lowest Delay Month: "
    f"{int(lowest_delay_month['order_month'])} "
    f"({lowest_delay_month['delay_pct']:.2f}%)"
)

print(
    f"Lowest Delay Day: "
    f"{lowest_delay_day['order_day']} "
    f"({lowest_delay_day['delay_pct']:.2f}%)"
)

print(
    f"Lowest Delay Hour: "
    f"{int(lowest_delay_hour['order_hour'])}:00 "
    f"({lowest_delay_hour['delay_pct']:.2f}%)"
)


# TIME BASED VISUALIZATION

fig, (ax1, ax2, ax3) = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


month_names = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec'
]


ax1.bar(
    delay_by_month['order_month'],
    delay_by_month['delay_pct'],
    color=primary_color
)

ax1.set_xticks(
    range(1, 13)
)

ax1.set_xticklabels(
    month_names,
    rotation=45
)

ax1.set_title(
    "Delay % by Month"
)

ax1.set_xlabel(
    "Month"
)

ax1.set_ylabel(
    "Delay Percentage (%)"
)


ax2.bar(
    delay_by_day['order_day'],
    delay_by_day['delay_pct'],
    color=secondary_color
)

ax2.set_title(
    "Delay % by Day"
)

ax2.set_xlabel(
    "Day of Week"
)

ax2.set_ylabel(
    "Delay Percentage (%)"
)

ax2.tick_params(
    axis='x',
    rotation=45
)


ax3.bar(
    delay_by_hour['order_hour'],
    delay_by_hour['delay_pct'],
    color=accent_color
)

ax3.set_title(
    "Delay % by Hour"
)

ax3.set_xlabel(
    "Order Hour"
)

ax3.set_ylabel(
    "Delay Percentage (%)"
)

ax3.set_xticks(
    range(0, 24, 2)
)


plt.tight_layout()

plt.show()


# PREDICTIVE MODELLING

print(
    "\n########## PREDICTIVE MODELLING #########"
)


# COPY DATA FOR MODELLING

model_df = df.copy()


# TARGET

y = (
    model_df['Is_Delayed']
    .astype(int)
)


# FEATURES

# These variables should not be used because they are
# created from the actual delivery outcome.

columns_to_remove = [
    'Is_Delayed',
    'Delay',
    'Delivery Status',
    'shipping date (DateOrders)',
    'order date (DateOrders)',
    'Profitability Flag',
    'Order Processing Time',
    'order_month',
    'order_day',
    'order_hour'
]


X = model_df.drop(
    columns=columns_to_remove,
    errors='ignore'
)


# REMOVE HIGHLY LEAKY DELIVERY VARIABLES

leakage_columns = [
    'Days for shipping (real)',
    'Late_delivery_risk'
]


X = X.drop(
    columns=leakage_columns,
    errors='ignore'
)


# REMOVE ROWS WITH MISSING VALUES

model_data = pd.concat(
    [X, y],
    axis=1
)

model_data = model_data.dropna()


X = model_data.drop(
    columns=['Is_Delayed']
)

y = model_data['Is_Delayed']


print(
    "\nModel Data Shape:"
)

print(
    model_data.shape
)


print(
    "\nTarget Distribution:"
)

print(
    y.value_counts()
)


# IDENTIFY DATA TYPES

numerical_columns = (
    X.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


categorical_columns = (
    X.select_dtypes(
        include='object'
    )
    .columns
    .tolist()
)


print(
    "\nNumerical Columns:"
)

print(
    numerical_columns
)


print(
    "\nCategorical Columns:"
)

print(
    categorical_columns
)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    "\nTraining Data:",
    X_train.shape
)

print(
    "Testing Data:",
    X_test.shape
)


# PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorical_columns
        )
    ],
    remainder='passthrough'
)


# LOGISTIC REGRESSION

logistic_model = Pipeline(
    steps=[
        (
            'preprocessor',
            preprocessor
        ),
        (
            'model',
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_prediction = (
    logistic_model
    .predict(X_test)
)


# LOGISTIC METRICS

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

logistic_precision = precision_score(
    y_test,
    logistic_prediction
)

logistic_recall = recall_score(
    y_test,
    logistic_prediction
)

logistic_f1 = f1_score(
    y_test,
    logistic_prediction
)


logistic_probability = (
    logistic_model
    .predict_proba(X_test)[:, 1]
)


logistic_auc = roc_auc_score(
    y_test,
    logistic_probability
)


print(
    "\n########## LOGISTIC REGRESSION RESULTS #########"
)

print(
    f"Accuracy: {logistic_accuracy:.4f}"
)

print(
    f"Precision: {logistic_precision:.4f}"
)

print(
    f"Recall: {logistic_recall:.4f}"
)

print(
    f"F1 Score: {logistic_f1:.4f}"
)

print(
    f"ROC-AUC: {logistic_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        logistic_prediction
    )
)


# RANDOM FOREST

random_forest_model = Pipeline(
    steps=[
        (
            'preprocessor',
            preprocessor
        ),
        (
            'model',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight=None
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_prediction = (
    random_forest_model
    .predict(X_test)
)


random_forest_probability = (
    random_forest_model
    .predict_proba(X_test)[:, 1]
)


# RANDOM FOREST METRICS

random_forest_accuracy = (
    accuracy_score(
        y_test,
        random_forest_prediction
    )
)

random_forest_precision = (
    precision_score(
        y_test,
        random_forest_prediction
    )
)

random_forest_recall = (
    recall_score(
        y_test,
        random_forest_prediction
    )
)

random_forest_f1 = (
    f1_score(
        y_test,
        random_forest_prediction
    )
)

random_forest_auc = (
    roc_auc_score(
        y_test,
        random_forest_probability
    )
)


print(
    "\n########## RANDOM FOREST RESULTS #########"
)

print(
    f"Accuracy: {random_forest_accuracy:.4f}"
)

print(
    f"Precision: {random_forest_precision:.4f}"
)

print(
    f"Recall: {random_forest_recall:.4f}"
)

print(
    f"F1 Score: {random_forest_f1:.4f}"
)

print(
    f"ROC-AUC: {random_forest_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        random_forest_prediction
    )
)


# BALANCED RANDOM FOREST USING SMOTE

print(
    "\n########## BALANCED RANDOM FOREST #########"
)


balanced_random_forest = ImbPipeline(
    steps=[
        (
            'preprocessor',
            preprocessor
        ),
        (
            'smote',
            SMOTE(
                random_state=42
            )
        ),
        (
            'model',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


balanced_random_forest.fit(
    X_train,
    y_train
)


balanced_prediction = (
    balanced_random_forest
    .predict(X_test)
)


balanced_probability = (
    balanced_random_forest
    .predict_proba(X_test)[:, 1]
)


balanced_accuracy = (
    accuracy_score(
        y_test,
        balanced_prediction
    )
)

balanced_precision = (
    precision_score(
        y_test,
        balanced_prediction
    )
)

balanced_recall = (
    recall_score(
        y_test,
        balanced_prediction
    )
)

balanced_f1 = (
    f1_score(
        y_test,
        balanced_prediction
    )
)

balanced_auc = (
    roc_auc_score(
        y_test,
        balanced_probability
    )
)


print(
    f"Accuracy: {balanced_accuracy:.4f}"
)

print(
    f"Precision: {balanced_precision:.4f}"
)

print(
    f"Recall: {balanced_recall:.4f}"
)

print(
    f"F1 Score: {balanced_f1:.4f}"
)

print(
    f"ROC-AUC: {balanced_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        balanced_prediction
    )
)


# MODEL COMPARISON

model_comparison = pd.DataFrame(
    {
        'Model': [
            'Logistic Regression',
            'Random Forest',
            'Balanced Random Forest'
        ],
        'Accuracy': [
            logistic_accuracy,
            random_forest_accuracy,
            balanced_accuracy
        ],
        'Precision': [
            logistic_precision,
            random_forest_precision,
            balanced_precision
        ],
        'Recall': [
            logistic_recall,
            random_forest_recall,
            balanced_recall
        ],
        'F1 Score': [
            logistic_f1,
            random_forest_f1,
            balanced_f1
        ],
        'ROC-AUC': [
            logistic_auc,
            random_forest_auc,
            balanced_auc
        ]
    }
)


print(
    "\n########## MODEL COMPARISON #########"
)

print(
    model_comparison
    .round(4)
    .to_string(index=False)
)


# MODEL COMPARISON VISUALIZATION

model_comparison_plot = (
    model_comparison
    .set_index('Model')
)


model_comparison_plot.plot(
    kind='bar',
    figsize=(12, 6),
    color=[
        primary_color,
        secondary_color,
        accent_color,
        neutral_color,
        danger_color
    ]
)


plt.title(
    "Model Performance Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.legend(
    loc='lower right'
)

plt.grid(
    True,
    linestyle=':',
    alpha=0.5
)

plt.tight_layout()

plt.show()


# CONFUSION MATRIX - RANDOM FOREST

random_forest_cm = confusion_matrix(
    y_test,
    random_forest_prediction
)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    random_forest_cm,
    annot=True,
    fmt='d',
    cmap='viridis',
    cbar=False
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.show()


# CONFUSION MATRIX - BALANCED RANDOM FOREST

balanced_cm = confusion_matrix(
    y_test,
    balanced_prediction
)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    balanced_cm,
    annot=True,
    fmt='d',
    cmap='viridis',
    cbar=False
)

plt.title(
    "Confusion Matrix - Balanced Random Forest"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.show()


# ROC CURVE

logistic_fpr, logistic_tpr, _ = (
    roc_curve(
        y_test,
        logistic_probability
    )
)


rf_fpr, rf_tpr, _ = (
    roc_curve(
        y_test,
        random_forest_probability
    )
)


balanced_fpr, balanced_tpr, _ = (
    roc_curve(
        y_test,
        balanced_probability
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression AUC = {logistic_auc:.3f}",
    color=primary_color
)


plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest AUC = {random_forest_auc:.3f}",
    color=secondary_color
)


plt.plot(
    balanced_fpr,
    balanced_tpr,
    label=f"Balanced RF AUC = {balanced_auc:.3f}",
    color=accent_color
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='gray'
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.grid(
    True,
    linestyle=':',
    alpha=0.5
)

plt.tight_layout()

plt.show()


# CROSS VALIDATION

print(
    "\n########## CROSS VALIDATION #########"
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


rf_cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)


balanced_rf_cv_scores = cross_val_score(
    balanced_random_forest,
    X_train,
    y_train,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)


print(
    "Random Forest F1 CV Scores:"
)

print(
    np.round(
        rf_cv_scores,
        4
    )
)


print(
    "Random Forest Mean CV F1:",
    round(
        rf_cv_scores.mean(),
        4
    )
)


print(
    "\nBalanced Random Forest F1 CV Scores:"
)

print(
    np.round(
        balanced_rf_cv_scores,
        4
    )
)


print(
    "Balanced Random Forest Mean CV F1:",
    round(
        balanced_rf_cv_scores.mean(),
        4
    )
)


# FEATURE IMPORTANCE

rf_preprocessor = (
    random_forest_model
    .named_steps['preprocessor']
)


rf_model = (
    random_forest_model
    .named_steps['model']
)


feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)


feature_importance = pd.DataFrame(
    {
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        'Importance',
        ascending=False
    )
)


print(
    "\n########## TOP PREDICTIVE FEATURES #########"
)

print(
    feature_importance
    .head(20)
    .round(4)
    .to_string(index=False)
)


# FEATURE IMPORTANCE VISUALIZATION

top_features = (
    feature_importance
    .head(15)
)


plt.figure(
    figsize=(10, 7)
)


sns.barplot(
    data=top_features,
    x='Importance',
    y='Feature',
    color=accent_color
)


plt.title(
    "Top 15 Features Predicting Late Delivery"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.grid(
    True,
    linestyle=':',
    alpha=0.5
)

plt.tight_layout()

plt.show()


# PREDICTED PROBABILITY

prediction_results = X_test.copy()


prediction_results['Actual_Delay'] = (
    y_test.values
)


prediction_results['Predicted_Delay'] = (
    balanced_prediction
)


prediction_results['Delay_Probability'] = (
    balanced_probability
)


print(
    "\n########## PREDICTION RESULTS #########"
)


print(
    prediction_results[
        [
            'Actual_Delay',
            'Predicted_Delay',
            'Delay_Probability'
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# RISK SEGMENTATION

prediction_results['Risk_Level'] = pd.cut(
    prediction_results['Delay_Probability'],
    bins=[
        0,
        0.30,
        0.70,
        1.00
    ],
    labels=[
        'Low Risk',
        'Medium Risk',
        'High Risk'
    ],
    include_lowest=True
)


print(
    "\n########## RISK SEGMENTATION #########"
)


risk_distribution = (
    prediction_results['Risk_Level']
    .value_counts()
    .sort_index()
)


print(
    risk_distribution
)


risk_percentage = (
    prediction_results['Risk_Level']
    .value_counts(normalize=True)
    .sort_index()
    * 100
)


print(
    "\nRisk Distribution (%):"
)

print(
    risk_percentage.round(2)
)


# RISK VISUALIZATION

plt.figure(
    figsize=(8, 5)
)


risk_percentage.plot(
    kind='bar',
    color=[
        primary_color,
        secondary_color,
        danger_color
    ]
)


plt.title(
    "Order Risk Distribution"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Percentage of Orders (%)"
)

plt.xticks(
    rotation=0
)

plt.grid(
    True,
    linestyle=':',
    alpha=0.5
)

plt.tight_layout()

plt.show()


# HIGH RISK ORDERS

high_risk_orders = prediction_results[
    prediction_results['Delay_Probability'] >= 0.70
]


print(
    "\n########## HIGH RISK ORDERS #########"
)


print(
    "High Risk Orders:",
    len(high_risk_orders)
)


print(
    "High Risk Orders %:",
    f"{len(high_risk_orders) / len(prediction_results) * 100:.2f}%"
)


# HIGH RISK ACTUAL DELAY RATE

if len(high_risk_orders) > 0:

    actual_high_risk_delay_rate = (
        high_risk_orders['Actual_Delay']
        .mean()
        * 100
    )

else:

    actual_high_risk_delay_rate = 0


print(
    "Actual Delay Rate Among High Risk Orders:",
    f"{actual_high_risk_delay_rate:.2f}%"
)


# PREDICTION PROBABILITY DISTRIBUTION

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    balanced_probability,
    bins=20,
    color=primary_color,
    edgecolor='white'
)


plt.axvline(
    0.70,
    color=danger_color,
    linestyle='--',
    linewidth=2,
    label='High Risk Threshold'
)


plt.xlabel(
    "Probability of Late Delivery"
)

plt.ylabel(
    "Number of Orders"
)

plt.title(
    "Predicted Probability of Late Delivery"
)

plt.legend()

plt.grid(
    True,
    linestyle=':',
    alpha=0.5
)

plt.tight_layout()

plt.show()


# ERROR ANALYSIS

prediction_results['Prediction_Correct'] = (
    prediction_results['Actual_Delay']
    == prediction_results['Predicted_Delay']
)


false_positives = prediction_results[
    (
        prediction_results['Actual_Delay'] == 0
    )
    &
    (
        prediction_results['Predicted_Delay'] == 1
    )
]


false_negatives = prediction_results[
    (
        prediction_results['Actual_Delay'] == 1
    )
    &
    (
        prediction_results['Predicted_Delay'] == 0
    )
]


print(
    "\n########## ERROR ANALYSIS #########"
)


print(
    "False Positives:",
    len(false_positives)
)


print(
    "False Negatives:",
    len(false_negatives)
)


print(
    "Total Incorrect Predictions:",
    (
        prediction_results['Prediction_Correct']
        == False
    ).sum()
)


# ACTUAL VS PREDICTED DISTRIBUTION

comparison = pd.DataFrame(
    {
        'Actual': y_test.value_counts(),
        'Predicted': pd.Series(
            balanced_prediction
        ).value_counts()
    }
)


comparison.index = [
    'On Time',
    'Delayed'
]


print(
    "\n########## ACTUAL VS PREDICTED #########"
)

print(
    comparison
)


# BUSINESS IMPACT BY RISK LEVEL

risk_business_analysis = (
    prediction_results
    .groupby('Risk_Level',
             observed=True)
    .agg(
        orders=(
            'Actual_Delay',
            'count'
        ),
        actual_delay_rate=(
            'Actual_Delay',
            'mean'
        ),
        average_predicted_probability=(
            'Delay_Probability',
            'mean'
        )
    )
    .reset_index()
)


risk_business_analysis[
    'actual_delay_rate'
] = (
    risk_business_analysis[
        'actual_delay_rate'
    ]
    * 100
)


print(
    "\n########## RISK LEVEL BUSINESS ANALYSIS #########"
)


print(
    risk_business_analysis
    .round(2)
    .to_string(index=False)
)


# FINAL MODEL SUMMARY

print(
    "\n########## PREDICTIVE MODELLING SUMMARY #########"
)


print(
    f"Logistic Regression Accuracy: "
    f"{logistic_accuracy:.4f}"
)


print(
    f"Random Forest Accuracy: "
    f"{random_forest_accuracy:.4f}"
)


print(
    f"Balanced Random Forest Accuracy: "
    f"{balanced_accuracy:.4f}"
)


print(
    f"Logistic Regression Precision: "
    f"{logistic_precision:.4f}"
)


print(
    f"Random Forest Precision: "
    f"{random_forest_precision:.4f}"
)


print(
    f"Balanced Random Forest Precision: "
    f"{balanced_precision:.4f}"
)


print(
    f"Logistic Regression Recall: "
    f"{logistic_recall:.4f}"
)


print(
    f"Random Forest Recall: "
    f"{random_forest_recall:.4f}"
)


print(
    f"Balanced Random Forest Recall: "
    f"{balanced_recall:.4f}"
)


print(
    f"Logistic Regression F1: "
    f"{logistic_f1:.4f}"
)


print(
    f"Random Forest F1: "
    f"{random_forest_f1:.4f}"
)


print(
    f"Balanced Random Forest F1: "
    f"{balanced_f1:.4f}"
)


print(
    f"Logistic Regression ROC-AUC: "
    f"{logistic_auc:.4f}"
)


print(
    f"Random Forest ROC-AUC: "
    f"{random_forest_auc:.4f}"
)


print(
    f"Balanced Random Forest ROC-AUC: "
    f"{balanced_auc:.4f}"
)


print(
    f"High Risk Orders: "
    f"{len(high_risk_orders)}"
)


print(
    f"High Risk Orders Percentage: "
    f"{len(high_risk_orders) / len(prediction_results) * 100:.2f}%"
)


print(
    "\nAnalysis completed successfully."
)
