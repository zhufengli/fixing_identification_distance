################################

#example usage for running one single treatment-effect pair on two ADMGs

################################
import fixing
import comparison_para as comparison
import fixing2prob as f2p

import numpy as np
import pandas as pd
from ananke import graphs

treatments = ["X1"]
outcomes = ["X3"]


vertices = ["X1", "X2", "X3","X4"]
gt_di = [("X1","X2"), ("X2","X3"), ("X1","X4"), ("X4","X3")]
gt_bi = [("X1","X3")]

G_gt = graphs.ADMG(vertices, gt_di, gt_bi)

identifier = fixing.identification.one_line.OneLineID(graph=G_gt, treatments=treatments, outcomes=outcomes)

solutions = identifier.functional()  
print(identifier.all_fix_orders, flush=True)
print(solutions)
print("Number of solutions found:", len(solutions))


all_fixing_gt = [None] * len(solutions)

for i, s in enumerate(solutions):
    expr = f2p.id_functional_expr(G_gt, s)
    all_fixing_gt[i] = expr


vertices = ["X1", "X2", "X3", "X4"]
pre_di = [("X1","X2"), ("X2","X3"), ("X4","X3"), ("X4","X2")]
pre_bi = [("X1","X3")]

G_pre = graphs.ADMG(vertices, pre_di, pre_bi)

identifier_pre = fixing.identification.one_line.OneLineID(graph=G_pre, treatments=treatments, outcomes=outcomes)

solutions_pre = identifier_pre.functional()  
print(identifier_pre.all_fix_orders, flush=True)
print(solutions_pre)
print("Number of solutions found:", len(solutions_pre))

all_fixing_pre = [None] * len(solutions_pre)

for i, s in enumerate(solutions_pre):
    expr = f2p.id_functional_expr(G_pre, s)
    all_fixing_pre[i] = expr

metrics = comparison.evaluate_prediction_lists(all_fixing_gt, all_fixing_pre)
print(metrics)

