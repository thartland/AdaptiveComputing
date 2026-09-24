import argparse
import logging
import numpy as np

from hf_simulation import func_hf 

def func_lf(x):
    return 0.5 * func_hf(x) + 10. * (x - 0.5) - 5.0
    #return 0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2) + (x - 0.5) * 10.0 - 5
    #return (x-3)**2

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
