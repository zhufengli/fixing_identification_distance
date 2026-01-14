from sympy import (
    symbols, Function, Wild, Integral, Sum, Tuple,
    together, srepr, oo
)
from sympy.core.function import AppliedUndef
import time
from multiprocessing import Pool, cpu_count 
import os
import sys, multiprocessing as mp

#if sys.platform == "darwin":          # macOS only
#    mp.set_start_method("fork", force=True)

P  = Function('P')      # joint P(x,y,…)
Pc = Function('Pc')     # conditional atom Pc(target , given)

def flat(obj):
    if isinstance(obj, (tuple, Tuple)):
        return sum((flat(el) for el in obj), ())
    return (obj,)

T, U = Wild('T'), Wild('U')

def _drop_marginals(node):
    if isinstance(node, (Sum, Integral)):
        inner = _drop_marginals(node.function)        # walk down first
        if (isinstance(inner, AppliedUndef)
            and inner.func is P                       # it really is P(...)
            and all(a == -oo and b == oo for _, a, b in node.limits)):
            bound = {v for v, _, _ in node.limits}    # variables in limits
            remainder = [x for x in inner.args if x not in bound]
            return P(*remainder) if remainder else 1
        return node.func(inner, *node.limits)
    elif node.is_Atom:
        return node
    else:
        return node.func(*(_drop_marginals(a) for a in node.args))


def fast_canon(expr):
    # 1. Bayes rule: Pc(target|given) → P(target,given)/P(given)
    expr = expr.replace(Pc(T, U),
                        lambda T, U: P(*flat(T), *flat(U)) / P(*flat(U)))

    # 2. Flatten towers of fractions
    expr = together(expr)

    # 3. Drop trivial summations / integrals of joints
    expr = _drop_marginals(expr)

    # 4. Canonical ordering of limits (optional but keeps strings unique)
    def _sort_limits(obj):
        if isinstance(obj, (Sum, Integral)):
            f = _sort_limits(obj.function)
            lim = tuple(sorted(obj.limits, key=lambda L: str(L[0])))
            return obj.func(f, *lim)
        elif obj.is_Atom:
            return obj
        return obj.func(*(_sort_limits(a) for a in obj.args))
    # Make the expression oder insensitive P(a,b) match P(b,a)
    def _sort_P_args(node):
        if isinstance(node, AppliedUndef) and node.func is P:
            # alphabetical (or srepr) order is enough for a unique representation
            return P(*sorted(node.args, key=lambda x: srepr(x)))
        elif node.is_Atom:
            return node
        return node.func(*(_sort_P_args(a) for a in node.args))
    
    expr = _sort_P_args(_sort_limits(expr))
    # 5. Return a cheap structural string
    return srepr(expr)

x,a,b,c,d,w = symbols('x a b c d w')


def evaluate_prediction_lists(list_ground_truth, list_predicted,
                              *, workers=None, _pool=None, _min_per_proc: int = 5):
    """
    Canonicalise both lists in parallel.

    Parameters
    ----------
    workers : int | None
        * None  → use all available logical CPUs  (default)
        * >=1   → use exactly that many processes
        * 1     → falls back to sequential evaluation
    """
    n_workers = cpu_count() if (workers is None) else max(int(workers), 1)

    total_items = len(list_ground_truth) + len(list_predicted)
    enough_work = total_items >= _min_per_proc * n_workers
    use_parallel = (_pool is not False) and (n_workers > 1) and enough_work

    if use_parallel:
        with Pool(processes=n_workers) as pool:
            canon_gt   = set(pool.map(fast_canon, list_ground_truth))
            canon_pred = set(pool.map(fast_canon, list_predicted))
    else:                                      # sequential fallback
        canon_gt   = {fast_canon(e) for e in list_ground_truth}
        canon_pred = {fast_canon(e) for e in list_predicted}

    n_pred, n_gt  = len(canon_pred), len(canon_gt)
    n_match       = len(canon_pred & canon_gt)

    p_correct_pred = n_match / n_pred if n_pred else 1.0
    p_covered_gt   = n_match / n_gt   if n_gt   else 1.0

    return {
        "predicted_correct"    : p_correct_pred,
        "predicted_incorrect"  : 1 - p_correct_pred,
        "ground_truth_covered" : p_covered_gt,
        "ground_truth_missing" : 1 - p_covered_gt,
        "canonlized_pred": n_pred,
        "canonlized_gt": n_gt,
        "overlap": n_match
    }
