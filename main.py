import argparse

import hmm
import htk


def init_argparse():
    parser = argparse.ArgumentParser("ZRE digit sequence recogniser")
    parser.add_argument('--phonemes', dest='phonemes', type=str, required=True)
    parser.add_argument('--dict', dest='dict', type=str, required=True)
    parser.add_argument('--input', dest='input', type=str, required=True)
    parser.add_argument('--frames', dest='frames', action='store_true')
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
    dictionary = read_dictionary(args.dict)
    phonemes = read_phonemes(args.phonemes)
    model = hmm.HMM(phonemes, dictionary)
    model.build_network()

    m = htk.readhtk(args.input)
    for d in m:
        model.step(d)
    model.print_result(args.frames)


if __name__ == "__main__":
    main()
