import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(args):
    # read in data
    df = pd.read_csv(args.input_csv)

    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df_train.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    df_test.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_csv", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    process_data(args)
