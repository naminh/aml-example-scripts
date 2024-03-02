import argparse
import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


def list_of_str(arg):
    return list(map(str, arg.split(',')))

def split_df_X_y(df, args):
    X = df[args.predictor_cols].copy()
    y = df[args.target_col].copy()

    return X, y

def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # eval
    eval_model(model, X_test, y_test)

    # return model
    return model

def eval_model(model, X_test, y_test):
    # make predictions
    y_pred = model.predict(X_test)

    # calculate evaluation metrics
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # print evaluation metrics
    print(classification_report(y_test, y_pred))
    print("ROC:", roc)
    print("F1:", f1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

def train(args):
    """
    Trains a machine learning model using the provided arguments.

    Args:
        args (Namespace): The command line arguments.

    Returns:
        None
    """
    # Start Auto Logging
    mlflow.start_run()
    mlflow.autolog()

    # setup parameters
    params = {
        "random_state": args.random_state,
    }

    # read in data
    train_df = pd.read_csv(os.path.join(args.train_data, "data.csv"))
    test_df = pd.read_csv(os.path.join(args.test_data, "data.csv"))

    # process data
    X_train, y_train = split_df_X_y(train_df, args)
    X_test, y_test = split_df_X_y(test_df, args)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )
    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.model, args.registered_model_name),
    )

    mlflow.end_run()

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--target_col", type=str)
    parser.add_argument("--predictor_cols", type=list_of_str)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model", type=str, help="folder path to model file")

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    train(args)
