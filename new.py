import argparse

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('-b', '--brains', help='brain type. Options:', type=str, nargs='+', default=['D3QN'])
args = parser.parse_args()

print(args.brains)