import numpy as np
import argparse

def build_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", dest="foo", type=float)

    return parser

def run(param_dict=None):
    if param_dict is None:
        parser = build_parser()
        param_dict = vars(parser.parse_args())

    return np.sin(param_dict["foo"])

if __name__ == "__main__":
    run()
