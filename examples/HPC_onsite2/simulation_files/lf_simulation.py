import argparse
import logging
import numpy as np

def func_lf(x):
    return (x-3)**2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LF synthetic objective simulation"
    )
    parser.add_argument(
        "--x",
        "-x",
        type=float,
        default=0.1,
        help="Input.",
    )
    args = parser.parse_args()
    performance = func_lf(args.x)
    print(f"Objective: {performance}")
