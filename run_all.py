"""Run the same computations as notebooks 00--05, in one fresh process."""
import argparse
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--demo',action='store_true',help='Use generated synthetic data only.')
    parser.add_argument('--source-root',help='Override the processed-data directory without editing the tracked configuration.')
    parser.add_argument('--run-name',help='Override the result subdirectory under outputs/.')
    args=parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(name,'4')
    from threadpoolctl import threadpool_limits
    from prelib.pipeline import run_all
    with threadpool_limits(limits=4):
        run_all(args.config,args.demo,source_root=args.source_root,run_name=args.run_name)


if __name__=='__main__':main()
