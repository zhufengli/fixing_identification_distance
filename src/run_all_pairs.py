################################

#example usage for running the comparison of two ADMGs

################################
import time
import tracemalloc
import fixing
import comparison_para as comparison
import fixing2prob as f2p

import numpy as np
import pandas as pd
from ananke import graphs

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run all graph pairs and report metrics."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=None,
        help=("Number of parallel worker processes "
              "(default: use all logical CPUs). Use 1 to disable parallelism.")
    )
    args = parser.parse_args()

    # Start profiling
    start_time = time.perf_counter()
    tracemalloc.start()

    def safe_metrics(gt_exprs, pre_exprs, id_gt, id_pre):
        """Return a metric dict given ground-truth & predicted expressions
           and their identifiability flags."""
        if id_gt and id_pre:
            return comparison.evaluate_prediction_lists(
                gt_exprs, pre_exprs, workers=args.workers
            )
        elif not id_gt and not id_pre:
            return dict(predicted_correct = 1.0,
                        predicted_incorrect = 0.0,
                        ground_truth_covered = 1.0,
                        ground_truth_missing = 0.0)
        else:
            return dict(predicted_correct = 0.0,
                        predicted_incorrect = 1.0,
                        ground_truth_covered = 0.0,
                        ground_truth_missing = 1.0)

    verts = ["X1", "X2", "X3", "X4", "X5"]

    gt_di = [("X1","X2"), ("X2","X3"), ("X1","X4"), ("X4","X3")]
    gt_bi = [("X1","X3")]

    pre_di = [("X1","X2"), ("X2","X3"), ("X4","X3"), ("X4","X2")]
    pre_bi = [("X1","X3")]

    G_gt  = graphs.ADMG(verts, gt_di,  gt_bi)
    G_pre = graphs.ADMG(verts, pre_di, pre_bi)

    metrics_pred_correct   = []   # predicted_correct
    metrics_pred_incorrect = []   # predicted_incorrect
    metrics_gt_covered     = []   # ground_truth_covered
    metrics_gt_missing     = []   # ground_truth_missing

    for treat in verts:
        for outcome in verts:
            if treat == outcome:
                continue
            treatments = [treat]
            outcomes   = [outcome]

            # ground-truth graph
            id_gt  = fixing.identification.one_line.OneLineID(
                         graph = G_gt, treatments = treatments,
                         outcomes = outcomes)
            if id_gt.id():                             
                sols_gt = id_gt.functional()
                exprs_gt = [f2p.id_functional_expr(G_gt, s) for s in sols_gt]
            else:
                exprs_gt = []                          

            # predicted graph
            id_pre = fixing.identification.one_line.OneLineID(
                         graph = G_pre, treatments = treatments,
                         outcomes = outcomes)
            if id_pre.id():
                sols_pre = id_pre.functional()
                exprs_pre = [f2p.id_functional_expr(G_pre, s) for s in sols_pre]
            else:
                exprs_pre = []

            m = safe_metrics(exprs_gt, exprs_pre, id_gt.id(), id_pre.id())

            metrics_pred_correct.append(m['predicted_correct'])
            metrics_pred_incorrect.append(m['predicted_incorrect'])
            metrics_gt_covered.append(m['ground_truth_covered'])
            metrics_gt_missing.append(m['ground_truth_missing'])

    print("predicted_correct :",   metrics_pred_correct)
    print("predicted_incorrect:",  metrics_pred_incorrect)
    print("ground_truth_covered:", metrics_gt_covered)
    print("ground_truth_missing:", metrics_gt_missing)

    def safe_mean(lst):
        return np.mean(lst) if lst else float("nan")

    avg_pred_correct   = safe_mean(metrics_pred_correct)
    avg_pred_incorrect = safe_mean(metrics_pred_incorrect)
    avg_gt_covered     = safe_mean(metrics_gt_covered)
    avg_gt_missing     = safe_mean(metrics_gt_missing)

    print("\n=== Averages across all pairs ===")
    print(f"predicted_correct   : {avg_pred_correct:.3f}")
    print(f"predicted_incorrect : {avg_pred_incorrect:.3f}")
    print(f"ground_truth_covered: {avg_gt_covered:.3f}")
    print(f"ground_truth_missing: {avg_gt_missing:.3f}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()

    print(f"\nElapsed time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: Current={current / 10**6:.2f} MB; Peak={peak / 10**6:.2f} MB")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
