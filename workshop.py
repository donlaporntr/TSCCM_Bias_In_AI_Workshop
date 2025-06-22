import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sns.set_theme(style="white", font_scale=1.2)

#########################################################################
# Utility functions                                                    #
#########################################################################

def get_dataframe (path) :
    df = pd.read_csv(path)
    df = make_gender_balance(df, 0.03, 0.01)
    return df

def Xy_combine (X_train, y_train, target='cirrhosis') : 
    df = X_train.copy()
    df[target] = y_train
    return df

def make_gender_balance(df, m_ratio, f_ratio):
    
    df = df.copy()

    group_M = df[df['gender'] == "M"]
    group_F = df[df['gender'] == "F"]

    M_1 = group_M[group_M['cirrhosis'] == 1]
    M_0 = group_M[group_M['cirrhosis'] == 0]
    F_1 = group_F[group_F['cirrhosis'] == 1]
    F_0 = group_F[group_F['cirrhosis'] == 0]

    # Set how many to sample
    M_total = min(len(M_1) / m_ratio, len(M_0) / (1-m_ratio))
    F_total = min(len(F_1) / f_ratio, len(F_0) / (1-f_ratio))

    # Compute sample sizes
    M_1_sample = int(M_total * m_ratio)
    M_0_sample = int(M_total * (1 - m_ratio))
    F_1_sample = int(F_total * f_ratio)
    F_0_sample = int(F_total * (1 - f_ratio))

    # Sample the data
    M_1_sampled = M_1.sample(n=M_1_sample, random_state=42)
    M_0_sampled = M_0.sample(n=M_0_sample, random_state=42)
    F_1_sampled = F_1.sample(n=F_1_sample, random_state=42)
    F_0_sampled = F_0.sample(n=F_0_sample, random_state=42)

    # Combine and shuffle
    combined_df = pd.concat([M_1_sampled, M_0_sampled, F_1_sampled, F_0_sampled]).sample(frac=1, random_state=42)

    return combined_df

#########################################################################
# Exploratory Analysis and Plots                                        #
#########################################################################

DISEASES = ["leukemia", "hepatic_failure", "immunosuppression", "lymphoma", "cirrhosis", "aids"]

def bar_chart_of_diseases(df, group_by, diseases=DISEASES): 
    long_df = pd.melt(
        df,
        id_vars=["patient_id", group_by],
        value_vars=diseases,
        value_name="presence",
        var_name="disease",
    )
    
    present_df = long_df[long_df["presence"] == 1]
    group_counts = df[group_by].value_counts()
    
    ratio_df = (
        present_df.groupby([group_by, "disease"]).size().reset_index(name="count")
    )
    ratio_df["ratio"] = ratio_df.apply(
        lambda row: row["count"] / group_counts[row[group_by]], axis=1
    )
    
    g = sns.catplot(
        data=ratio_df,
        x="disease",
        y="ratio",
        col=group_by,
        col_wrap=3,
        hue="disease",
        kind="bar",
        aspect=2,
        legend=False,
    )

    if len(diseases) > 3 :
        for ax in g.axes.flatten():
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    g.set_titles(col_template="{col_name}")
    if len(diseases) == 1: 
        g.set_axis_labels("", "Ratio of Patients")
    else:
        g.set_axis_labels("Disease", "Ratio of Patients")
    plt.tight_layout()
    plt.show()

def bar_chart_of_cirrhosis (df, group_by): 
    return bar_chart_of_diseases(df, group_by, ["cirrhosis"])

def display_feature_columns (df, n_cols = 4, spacing = 3) :
    cols = [x for x in df.columns if x.startswith('d1_') and x.endswith('_min') ]
    cols = [x.replace('d1_', '').replace('_min', '') for x in cols]

    length = max(len(x) for x in cols) + spacing
               
    remainder = len(cols) % n_cols
    if remainder != 0:
        cols += [""] * (n_cols - remainder)
    rows = [cols[i:i + n_cols] for i in range(0, len(cols), n_cols)]
    for row in rows:
        print(" ".join(["{:<" + str(length) + "}"] * n_cols).format(*row))


#########################################################################
# Training ML Model                                                     #
#########################################################################

def evaluate(model, X_test, y_test, group_test):
    y_pred = model.predict(X_test)

    result_df = pd.DataFrame({
        "ground_truth": y_test,
        "prediction": y_pred,
    })
    result_df["TP"] = (result_df["ground_truth"] == 1) & (result_df["prediction"] == 1)
    result_df["TN"] = (result_df["ground_truth"] == 0) & (result_df["prediction"] == 0)
    result_df["FP"] = (result_df["ground_truth"] == 0) & (result_df["prediction"] == 1)
    result_df["FN"] = (result_df["ground_truth"] == 1) & (result_df["prediction"] == 0)

    result_df = pd.concat([result_df, group_test], axis=1)
    return result_df

def train(
    X_train, 
    y_train,
    numerical_features,
    categorical_features
): 
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline([
        ("transformer", transformer),
        # Add our classifier.
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            min_samples_split=10,
        ))
    ])

    model.fit(X_train, y_train)
    return model

def make_balance (X_train, y_train):
    df = X_train.copy()
    df['target'] = y_train

    group_true = df[df['target'] == True]
    group_false = df[df['target'] == False]

    min_size = min(len(group_true), len(group_false))

    group_true_sampled = group_true.sample(n=min_size, random_state=42)
    group_false_sampled = group_false.sample(n=min_size, random_state=42)

    balanced_df = pd.concat([group_true_sampled, group_false_sampled]).sample(frac=1, random_state=42)

    y_balanced = balanced_df['target']
    X_balanced = balanced_df.drop(columns=['target'])

    return X_balanced, y_balanced

def train_and_evaluate(
    df,
    numerical_features: list[str],
    categorical_features: list[str],
    target: str,
    groups: list[str] = ["gender", "age", "ethnicity"],
    everyone_as_male: bool = False,
    mitigate_bias: bool = None,
):

    df = df.dropna(subset=[target]).copy()
    features = numerical_features + categorical_features

    X = df[features]
    y = df[target]
    group = df[groups]

    X_train, X_test, y_train, y_test, _, group_test = train_test_split(
        X, y, group, test_size=0.3, random_state=42
    )

    if mitigate_bias is not None: 
        X_train, y_train = mitigate_bias(X_train, y_train)

    X_train, y_train = make_balance(X_train, y_train)    
    model = train(X_train, y_train, numerical_features, categorical_features)
    features = categorical_features + numerical_features

    if everyone_as_male and "gender" in features: 
        X_test["gender"] = "M"

    return evaluate(model, X_test, y_test, group_test)

def compute_metrics(group):
    tp = group["TP"].sum()
    tn = group["TN"].sum()
    fp = group["FP"].sum()
    fn = group["FN"].sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return pd.Series({"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1})

def plot_model_result(data: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    kwargs = {"annot": True, "vmin": 0, "vmax": 1, "cmap": "YlGnBu", "fmt": ".2f"}

    display_columns = ["TP", "FP", "TN", "FN"]

    # Raw count heatmaps
    sns.heatmap(data=data.groupby("gender")[display_columns].sum(), ax=axes[0], annot=True)
    axes[0].set_title("Confusion Counts by Gender")

    # Metrics heatmaps
    gender_metrics = data.groupby("gender").apply(compute_metrics, include_groups=False)

    sns.heatmap(data=gender_metrics, ax=axes[1], **kwargs)
    axes[1].set_title("Metrics by Gender")

    plt.tight_layout()
    plt.show()
