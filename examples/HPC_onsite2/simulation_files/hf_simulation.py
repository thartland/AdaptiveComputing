import argparse
import logging
import numpy as np

def func_hf(x):
    return (x-3)**2 + 0.1 * np.sin(x)

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
    performance = func_hf(args.x)
    print(f"Objective: {performance}")
