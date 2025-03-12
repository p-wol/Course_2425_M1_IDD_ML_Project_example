import argparse
import os
import pickle
import json
import numpy as np
import scipy
import polars as pl
from time import gmtime, strftime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge

# Arguments
parser = argparse.ArgumentParser(
        prog = "24/25 ML Project Example",
        description = "Example program for the ML Project course (2024/2025 M1 IDD)")

parser.add_argument("--dataset_path", type = str, default = "", help = "path to the dataset file")
parser.add_argument("--ml_method", type = str, default = "Linear", help = "name of the ML method to use ('Linear', 'Poisson')")
parser.add_argument("--l2_penalty", type = float, default = 1., help = "strength of the L2 penalty used when fitting the model")
parser.add_argument("--cv_nsplits", type = int, default = 5, help = "cross-validation: number of splits")
parser.add_argument("--save_dir", type = str, default = "", help = "where to save the model, the logs and the configuration")

args = parser.parse_args()

# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")

# Store the configuration
with open(path_config, "w") as f:
    json.dump(vars(args), f)

# Loading the dataset
df = pl.read_csv(args.dataset_path)

# Build the features and the targets
columns = ["mnth", "hr", "workingday", "weathersit", "temp"]

X = df[columns].to_pandas()
y = df["bikers"].to_pandas()

# Preprocessing
lst_categ = [["Jan", "Feb", "March", "April", "May", "June", 
    "July", "Aug", "Sept", "Oct", "Nov", "Dec"],
    list(range(24)),
    [0, 1],
    df["weathersit"].unique().to_numpy()]

preprocess = ColumnTransformer([("ohe", OneHotEncoder(categories = lst_categ), ["mnth", "hr", "workingday", "weathersit"]),
                                ("identity", FunctionTransformer(), ["temp"])])

# Build the model
if args.ml_method == "Linear":
    model = Ridge(alpha = args.l2_penalty)
elif args.ml_method == "Poisson":
    model = PoissonRegressor(alpha = args.l2_penalty)
else:
    raise ValueError(f"Invalid value found for argument 'ml_method': found '{args.ml_method}'")

# Build the pipeline
pipeline = Pipeline([("preprocess", preprocess), ("model", model)])

# Fit the data
pipeline.fit(X, y)

# Save model
with open(path_model, 'wb') as f:
    pickle.dump({"model": model}, f)

# Test model
lst_scores_mse = cross_val_score(pipeline, X, y, cv = args.cv_nsplits, scoring = "neg_mean_squared_error")
score_mse = sum(lst_scores_mse) / args.cv_nsplits

lst_scores_r2 = cross_val_score(pipeline, X, y, cv = args.cv_nsplits)
score_r2 = sum(lst_scores_r2) / args.cv_nsplits

# Store results
with open(path_logs, "w") as f:
    json.dump({"score_mse": score_mse,
        "score_r2": score_r2}, f)

