import argparse
import data

def threshold(graph, pattern, error, ns=None, c=None, cores=1):
    """Return ``GS`` or ``NS`` for one supplied workload."""
    raise NotImplementedError("workload selector removed")

def main(graph, pattern, error, ns, c, cores):
    print(threshold(graph, pattern, error, ns, c, cores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for different modes")
    parser.add_argument('mode', choices=['threshold'], help="Select threshold mode")
    parser.add_argument('--graph', type=str, help="Specify the graph string")
    parser.add_argument('--pattern', type=str, help="Specify the pattern string")
    parser.add_argument('--error', type=float, help="Acceptable Error Bound (0.0 - 1.0)")
    parser.add_argument('--NS', type=int, help="Specify NS (optional)")
    parser.add_argument('--C', type=float, help="Specify C for graph sparsification (optional)")
    parser.add_argument('--cores', type=int, default=1, help="Available CPU cores")

    args = parser.parse_args()
    main(args.graph, args.pattern, args.error,args.NS, args.C, args.cores)
