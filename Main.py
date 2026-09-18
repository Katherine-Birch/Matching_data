'''
Main file to run
'''
import argparse
from phase1_data_prep import run_phase_1
from phase2_benchmarking import run_phase_2
from phase3_scaling_transfer import run_phase_3
from phase4_synthetic_alignment import run_phase_4

def main():
    parser = argparse.ArgumentParser(description="Graph Matching Benchmark Pipeline")
    parser.add_argument('--run-phase', type=int, choices=[1, 2, 3, 4], help="Run a specific pipeline phase")
    parser.add_argument('--run-all', action='store_true', help="Run all phases sequentially")
    args = parser.parse_args()

    if args.run_phase == 1 or args.run_all:
        run_phase_1()
    if args.run_phase == 2 or args.run_all:
        run_phase_2()
    if args.run_phase == 3 or args.run_all:
        run_phase_3()
    if args.run_phase == 4 or args.run_all:
        run_phase_4()

if __name__ == "__main__":
    main()