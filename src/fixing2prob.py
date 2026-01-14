import re
import networkx as nx
from ananke.graphs import ADMG
from sympy import Function, Sum, Integral, Tuple, symbols, oo, Mul
from functools import reduce

# ── probability atoms that fast_canon() already understands ─────────
P  = Function("P")       # joint
# no conditional from the joint – we build them ourselves

# ── helper functions ----------------------------------------------------------
def _T(it):
    return Tuple(*sorted(it, key=str)) if it else Tuple()

def _Σ(expr, var, discrete=True):
    sym            = symbols(var)
    indep, depend  = expr.as_independent(sym)
    op             = Sum if discrete else Integral
    return indep * op(depend, (sym, -oo, oo))

def _directed_part(admg: ADMG):
    """Return a DiGraph with all vertices"""
    g = nx.DiGraph()
    g.add_nodes_from(admg.vertices)
    g.add_edges_from(admg.di_edges)
    return g

def _non_descendants(G, v):
    return set(G.nodes) - nx.descendants(G, v) - {v}

def _children(G, v):
    return set(G.successors(v))

def _conditional_from_kernel(q, target, given, vars_all, *, discrete=True):
    """
    q(target,given)/q(given)  where both numer & denom are obtained from q by
    summing/integrating out the complementary variables.
    """
    rest_numer = [x for x in vars_all if x not in target and x not in given]
    rest_denom = [x for x in vars_all if x not in given]

    numer = q
    for x in rest_numer:
        numer = _Σ(numer, x, discrete)

    denom = q
    for x in rest_denom:
        denom = _Σ(denom, x, discrete)

    return numer / denom

# ── Φ-string parsers -------------------------------------------------
def parse_phi_sequence(phi_str):
    m = re.search(r"Φ([^Σ\(]+?)\(", phi_str.replace(" ", ""))
    if not m:
        raise ValueError("no Φ-sequence found")
    seq = re.findall(r"X\d+", m.group(1))
    return list(reversed(seq))  


def extract_phi_sequences(one_line_id_output):
    """Return list of sequences (each itself a list of variables)."""
    return [parse_phi_sequence(tok)
            for tok in one_line_id_output.split()
            if tok.lstrip("Σ").startswith("Φ")]

# ── main routine -----------------------------------------------------
def fixing_kernel_expr(admg: ADMG,
                       phi,
                       *,
                       discrete=True,
                       joint_order=None):
    """
    Parameters
    ----------
    admg  :  Ananke ADMG (directed & bidirected edges)
    phi   :  list/tuple with the variables to fix *in that order*
    Returns
    -------
    SymPy expression   – kernel after the whole Φ-sequence
    """
    Gdir = _directed_part(admg)          

    if joint_order is None:
        joint_order = list(nx.topological_sort(Gdir))

    q   = P(_T(joint_order))
    var = list(joint_order)             

    for v in phi:
        childset = _children(Gdir, v)

        if childset:                                  
            ND   = _non_descendants(Gdir, v)
            cond = _conditional_from_kernel(q,
                                            target=[v],
                                            given=ND,
                                            vars_all=var,
                                            discrete=discrete)
            q = q / cond                           

        if not childset:
            q = _Σ(q, v, discrete)                    
            var.remove(v)                             

        Gdir.remove_edges_from(list(Gdir.in_edges(v)))

    return q


# ── Σ-parsers ---------------------------------------------------------
def _sigma_vars(tok):
    return re.findall(r"X\d+", tok)

def extract_sigma_vars(one_line_str):
    """Return Σ-variables in the order they appear in the One-Line string."""
    out = []
    for tok in one_line_str.replace("(", " ").split():
        if tok.startswith("Σ"):
            out += _sigma_vars(tok)
    return out

# ── top-level ---------------------------------------------------------
def id_functional_expr(admg,
                        one_line_id,
                        *,
                        discrete=True,
                        joint_order=None):
    """
    Parameters
    ----------
    admg         : Ananke ADMG
    one_line_id  : string produced by ananke.identification.one_line.OneLineID
    Returns
    -------
    SymPy expression containing P, Sum/Integral, Tuple – ready for fast_canon
    """
    # 1. parse
    sigma_vars   = extract_sigma_vars(one_line_id)
    phi_sequences = extract_phi_sequences(one_line_id)

    # 2. kernels for every Φ
    kernels = [
        fixing_kernel_expr(admg,
                           phi,
                           discrete=discrete,
                           joint_order=joint_order)
        for phi in phi_sequences
    ]

    if not kernels:
        raise ValueError("No Φ-sequence found – nothing to compute.")

    expr = reduce(Mul, kernels)              # product of all kernels

    # 3. wrap in Σ / ∫  (keep order)
    for v in sigma_vars:
        expr = _Σ(expr, v, discrete)

    return expr