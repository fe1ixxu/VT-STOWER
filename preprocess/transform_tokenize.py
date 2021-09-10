from transformers import AutoTokenizer, AutoModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    parser.add_argument('--pretrained_model', type=str, required=True, help='pretrained language model')  
    parser.add_argument('--suffix', type=int, default=None, help='suffix')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    fo = open(args.input, encoding="utf-8")
    fw = open(args.output, "w", encoding="utf-8")
    line = fo.readline()
    while(line):
        line = line.strip()
        toks = tokenizer.tokenize(line)
        if args.suffix == 0:
            suffix = [tokenizer.bos_token]
        elif args.suffix == 1:
            suffix = [tokenizer.eos_token]
        else:
            suffix = []

        toks = " ".join(toks + suffix)

        fw.writelines([toks, "\n"])
        line = fo.readline()

    fo.close()
    fw.close()

if __name__ == "__main__":
  main()
