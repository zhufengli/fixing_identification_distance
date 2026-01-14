import numpy as np
import pandas as pd
from ananke import graphs
from ananke import identification
from ananke.identification.one_line import NotIdentifiedError
import matplotlib.pyplot as plt

import copy

_OriginalFixable = graphs.ADMG.fixable

def fixable_all(self, vertices):
    """
    A patched version of fixable() that returns ALL valid fix orders
    Returns:
        (bool, list_of_orders)
           - bool = True if there's at least one valid fix order,
                    False otherwise
           - list_of_orders = a list of valid fix orders (each is a list of variable names)
    """

    vertices = set(vertices) - set(self.fixed)
    if not vertices:
        return (True, [[]])

    results = []

    def backtrack(graph, remaining, current_order):
        if not remaining:
            results.append(current_order[:])  
            return

        fixable_now = []
        for v in list(remaining):
            if len(graph.descendants([v]).intersection(graph.district(v))) == 1:
                fixable_now.append(v)

        if not fixable_now:
            return

        for v in fixable_now:
            G_copy = copy.deepcopy(graph)
            G_copy.fix([v])

            current_order.append(v)
            backtrack(G_copy, remaining - {v}, current_order)
            current_order.pop()

    G_copy = copy.deepcopy(self)
    backtrack(G_copy, vertices, [])

    return (len(results) > 0, results)

graphs.ADMG.fixable = fixable_all


_OriginalID = identification.one_line.OneLineID.id

def id_all(self):
    """
    Patched id() that collects ALL valid fix orders for each district
    in Gystar, instead of just storing one.
    
    Returns True if identified (at least one fix order works for each district),
    otherwise False.
    """
    self.fixing_orders = {}     
    self.all_fix_orders = {}     

    vertices = set(self.graph.vertices)

    # Check each district in Gystar
    for district in self.Gystar.districts:
        fixable, orders = self.graph.fixable(vertices - district)
        if not fixable:
            return False  # if we can't fix (V \ D) for this district => not ID

        # Store them
        self.fixing_orders[tuple(district)] = orders[0]  # pick one arbitrarily
        self.all_fix_orders[tuple(district)] = orders

    return True

identification.one_line.OneLineID.id = id_all

import itertools

_OriginalFunctional = identification.one_line.OneLineID.functional

def functional_all(self):
    """
    Patched functional() that returns a LIST of factorization strings,
    one for each valid combination of fix orders across all districts.
    """
    if not self.id():
        raise NotIdentifiedError("Not identifiable")

    if not hasattr(self, "all_fix_orders") or len(self.all_fix_orders) == 0:
        return []

    if set(self.ystar) == set(self.outcomes):
        prefix = ""
    else:
        prefix = "\u03A3" 
        hidden_vars = [y for y in self.ystar if y not in self.outcomes]
        prefix += "".join(hidden_vars)
        if len(self.ystar) > 1:
            prefix += " "

    districts = sorted(self.all_fix_orders.keys(), key=lambda d: tuple(d))

    all_factorizations = []

    list_of_lists = [self.all_fix_orders[d] for d in districts]

    for combo_of_orders in itertools.product(*list_of_lists):
        expr = prefix
        for district_idx, d in enumerate(districts):
            fix_order = combo_of_orders[district_idx]
            expr += "\u03A6" + "".join(reversed(fix_order)) + "(p(V);G) "

        all_factorizations.append(expr.strip())

    return all_factorizations

identification.one_line.OneLineID.functional = functional_all