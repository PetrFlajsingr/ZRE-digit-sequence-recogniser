import argparse

import htk


def init_argparse():
    parser = argparse.ArgumentParser("ZRE digit sequence recogniser")
    return parser

def main():
    args = init_argparse().parse_args()
    m = htk.readhtk('./dev/a30000b1.lik')
    print(m)

if __name__ == "__main__":
    main()
