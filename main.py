import argparse

import hmm
import htk


def init_argparse():
    parser = argparse.ArgumentParser("ZRE digit sequence recogniser")
    return parser


def read_dictionary(path: str):
    fp = open(path, encoding='ISO8859-2')
    result = []
    try:
        for line in fp:
            line = line.strip()
            split = line.split('\t')
            word = split[0]
            phonemes = split[1].split(' ')
            result.append((word, phonemes))
    finally:
        fp.close()
    return result


def read_phonemes(path: str):
    fp = open(path, encoding='ISO8859-2')
    result = []
    try:
        for line in fp:
            line = line.strip()
            result.append(line)
    finally:
        fp.close()
    return result


def main():
    args = init_argparse().parse_args()
    dictionary = read_dictionary('./dicos/zre.dict')
    phonemes = read_phonemes('./dicos/phonemes')
    model = hmm.HMM(phonemes, dictionary)
    model.build_network()

    m = htk.readhtk('./dev/a30000b1.lik')
    for d in m:
        model.step(d)

    print("done")
    model.print_result()

    model.print()


if __name__ == "__main__":
    main()
