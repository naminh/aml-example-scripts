import argparse
import os
import pyspark.pandas as pd
from pyspark.sql import SparkSession



def spark_step(args):
    print("Starting Spark step...")
    sc = SparkSession.builder.getOrCreate()
    df = pd.read_csv(args.input_uri)

    df_sampled = df.sample(frac=0.9)
    df_sampled.to_csv(args.output)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_uri", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--index_col", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    spark_step(args)
