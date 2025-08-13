# -*- coding: utf-8 -*-
"""
AFN por Thompson + dibujo (DOT/Graphviz) + simulación interactiva.
- Estados numerados 0..n (0 es el inicial).
- Inicial con borde verde + flecha desde punto.
- Aceptación con doble círculo.
- Transiciones completas (incluye ε).
- Leyenda integrada.
- Dibujo LR con DOT; salida PNG en UTF-8.

Uso:
    python thompson_nfa.py entrada.txt
"""

import sys
import re
import itertools
import subprocess
import shutil
from collections import defaultdict, deque

try:
    import pydot  # Construimos el DOT con pydot, render con 'dot'
except Exception as e:
    print("Este script requiere pydot. Instala con: pip install pydot")
    raise

# ===============================
# Parsing a Postfijo (Shunting-yard)
# ===============================
OPERATORS = {'|', '.', '*', '+', '?'}
PREC = {'|': 1, '.': 2, '*': 3, '+': 3, '?': 3}
RIGHT_ASSOC = {'*', '+', '?'}

def normalize_regex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace('∗', '*')  # estrella unicode -> ascii
    return s

def is_literal(ch: str) -> bool:
    return ch not in OPERATORS and ch not in {'(', ')'}

def insert_concat_dots(regex: str) -> str:
    out = []
    for i, a in enumerate(regex):
        out.append(a)
        if i + 1 >= len(regex):
            continue
        b = regex[i+1]
        if (is_literal(a) or a == ')' or a in {'*', '+', '?'}) and (is_literal(b) or b == '('):
            out.append('.')
    return ''.join(out)

def to_postfix(regex: str) -> str:
    output, stack = [], []
    for ch in regex:
        if is_literal(ch) or ch == 'ε':
            output.append(ch)
        elif ch in OPERATORS:
            while stack:
                top = stack[-1]
                if top in OPERATORS and (
                    (PREC[top] > PREC[ch]) or
                    (PREC[top] == PREC[ch] and ch not in RIGHT_ASSOC)
                ):
                    output.append(stack.pop())
                else:
                    break
            stack.append(ch)
        elif ch == '(':
            stack.append(ch)
        elif ch == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack:
                raise ValueError("Paréntesis desbalanceados")
            stack.pop()
        else:
            raise ValueError(f"Carácter no reconocido: {ch}")
    while stack:
        op = stack.pop()
        if op in {'(', ')'}:
            raise ValueError("Paréntesis desbalanceados")
        output.append(op)
    return ''.join(output)

# ===============================
# Thompson: construcción de AFN
# ===============================
class State:
    _ids = itertools.count()
    def __init__(self):
        self.id = next(State._ids)
        self.trans = defaultdict(set)  # símbolo -> set(State)
        self.eps = set()               # epsilon -> set(State)
    def add_edge(self, symbol, target):
        if symbol == 'ε':
            self.eps.add(target)
        else:
            self.trans[symbol].add(target)

class Fragment:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

    # ---------- Simulación ----------
    def epsilon_closure(self, states):
        stack = list(states)
        closure = set(states)
        while stack:
            s = stack.pop()
            for t in s.eps:
                if t not in closure:
                    closure.add(t)
                    stack.append(t)
        return closure

    def move(self, states, symbol):
        out = set()
        for s in states:
            for t in s.trans.get(symbol, []):
                out.add(t)
        return out

    def accepts(self, w: str) -> bool:
        current = self.epsilon_closure({self.start})
        for ch in w:
            current = self.epsilon_closure(self.move(current, ch))
            if not current:
                break
        return self.accept in current

def thompson_from_postfix(postfix: str) -> NFA:
    stack = []
    for tok in postfix:
        if tok not in OPERATORS or tok == 'ε':
            s, t = State(), State()
            if tok == 'ε':
                s.add_edge('ε', t)
            else:
                s.add_edge(tok, t)
            stack.append(Fragment(s, t))

        elif tok == '.':
            b, a = stack.pop(), stack.pop()
            a.accept.add_edge('ε', b.start)
            stack.append(Fragment(a.start, b.accept))

        elif tok == '|':
            b, a = stack.pop(), stack.pop()
            s, t = State(), State()
            s.add_edge('ε', a.start); s.add_edge('ε', b.start)
            a.accept.add_edge('ε', t); b.accept.add_edge('ε', t)
            stack.append(Fragment(s, t))

        elif tok == '*':
            a = stack.pop()
            s, t = State(), State()
            s.add_edge('ε', a.start); s.add_edge('ε', t)
            a.accept.add_edge('ε', a.start); a.accept.add_edge('ε', t)
            stack.append(Fragment(s, t))

        elif tok == '+':
            a = stack.pop()
            s, t = State(), State()
            s.add_edge('ε', a.start)             # al menos una vez
            a.accept.add_edge('ε', a.start)      # repetir
            a.accept.add_edge('ε', t)            # salida
            stack.append(Fragment(s, t))

        elif tok == '?':
            a = stack.pop()
            s, t = State(), State()
            s.add_edge('ε', a.start); s.add_edge('ε', t)
            a.accept.add_edge('ε', t)
            stack.append(Fragment(s, t))

        else:
            raise ValueError(f"Token no soportado: {tok}")

    if len(stack) != 1:
        raise ValueError("Expresión mal formada")
    frag = stack.pop()
    return NFA(frag.start, frag.accept)

def build_nfa(regex_infix: str) -> NFA:
    r = normalize_regex(regex_infix)
    r = insert_concat_dots(r)
    postfix = to_postfix(r)
    return thompson_from_postfix(postfix)

# ===============================
# Estados y dibujo (DOT/Graphviz)
# ===============================
def enumerate_states(start: State):
    """BFS para listar todos los estados alcanzables desde start."""
    seen, order, q = set(), [], deque([start])
    while q:
        s = q.popleft()
        if s in seen:
            continue
        seen.add(s); order.append(s)
        for t in s.eps:
            if t not in seen: q.append(t)
        for ts in s.trans.values():
            for t in ts:
                if t not in seen: q.append(t)
    return order

def relabel_dense(nfa: NFA):
    """Reindexa estados como 0,1,... en orden BFS (0 es el start)."""
    states = enumerate_states(nfa.start)
    id_map = {s: i for i, s in enumerate(states)}
    return id_map, states

def build_pydot_graph(nfa: NFA) -> pydot.Dot:
    """Construye el DOT con:
       - nodos etiquetados 0..n,
       - inicial borde verde,
       - aceptación doble círculo,
       - leyenda,
       - transiciones completas (incluye ε)."""
    id_map, states = relabel_dense(nfa)

    G = pydot.Dot(
        graph_type="digraph",
        rankdir="LR",
        splines="spline",
        nodesep="0.45",
        ranksep="0.75",
        concentrate="false",
        charset="UTF-8"
    )
    # Fuente con soporte Unicode
    G.set_node_defaults(shape="circle", fontsize="12", fontname="DejaVu Sans", color="black")
    G.set_edge_defaults(fontsize="11", fontname="DejaVu Sans", arrowsize="0.8")

    # Nodo de inicio puntual
    G.add_node(pydot.Node("__START__", shape="point", width="0.1", color="green", label=""))

    # Nodos de estados (0..n)
    for s in states:
        name = str(id_map[s])     # nombre del nodo y etiqueta numérica
        label = name
        is_start = (s is nfa.start)
        is_accept = (s is nfa.accept)

        attrs = {"label": label}
        if is_accept:
            attrs["shape"] = "doublecircle"
            attrs["penwidth"] = "2.2"
        if is_start:
            # resaltar inicial con borde verde (además de la flecha desde __START__)
            attrs["color"] = "green"
            attrs["penwidth"] = attrs.get("penwidth", "2.0")

        G.add_node(pydot.Node(name, **attrs))

    # Transiciones (ε y no-ε)
    edge_labels = defaultdict(set)
    for s in states:
        for t in s.eps:
            edge_labels[(id_map[s], id_map[t])].add('ε')
        for sym, ts in s.trans.items():
            for to in ts:
                edge_labels[(id_map[s], id_map[to])].add(sym)

    for (u, v), labels in edge_labels.items():
        lbl = ",".join(sorted(labels))  # p.ej. "a,b,ε"
        G.add_edge(pydot.Edge(str(u), str(v), label=lbl))

    # Flecha de inicio a 0
    G.add_edge(pydot.Edge("__START__", str(id_map[nfa.start]), label=""))

    # ========= LEYENDA =========
    legend = pydot.Cluster(graph_name="cluster_leyenda", label="Leyenda", fontsize="12",
                           fontname="DejaVu Sans", color="gray50", style="rounded")
    # Fila: Inicial
    legend.add_node(pydot.Node("leg_start_icon", shape="point", width="0.12", color="green", label=""))
    legend.add_node(pydot.Node("leg_start_text", shape="plaintext", label="Inicial"))
    legend.add_edge(pydot.Edge("leg_start_icon", "leg_start_text", style="invis"))
    # Fila: Adicional
    legend.add_node(pydot.Node("leg_other_icon", shape="circle", label=""))
    legend.add_node(pydot.Node("leg_other_text", shape="plaintext", label="Adicional"))
    legend.add_edge(pydot.Edge("leg_other_icon", "leg_other_text", style="invis"))
    # Fila: Aceptación
    legend.add_node(pydot.Node("leg_accept_icon", shape="doublecircle", penwidth="2.2", label=""))
    legend.add_node(pydot.Node("leg_accept_text", shape="plaintext", label="Aceptación"))
    legend.add_edge(pydot.Edge("leg_accept_icon", "leg_accept_text", style="invis"))

    G.add_subgraph(legend)
    # ===========================

    return G

def render_png_with_dot(G: pydot.Dot, out_png: str):
    """
    Renderiza a PNG llamando a 'dot' por subprocess, alimentándole el DOT en UTF-8.
    """
    dot_exe = shutil.which("dot")
    if not dot_exe:
        raise RuntimeError("Graphviz 'dot' no está en PATH.")

    dot_src: str = G.to_string()  # cadena DOT (incluye ε)
    proc = subprocess.run(
        [dot_exe, "-Tpng", "-o", out_png],
        input=dot_src.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot falló:\n{proc.stderr.decode('utf-8', errors='replace')}")

def draw_nfa(nfa: NFA, filename_png: str):
    G = build_pydot_graph(nfa)
    render_png_with_dot(G, filename_png)

# ===============================
# Main (interactivo: pide w por cada regex)
# ===============================
def main():
    if len(sys.argv) < 2:
        print("Uso: python thompson_nfa.py <archivo_regex_por_linea.txt>")
        sys.exit(1)

    in_file = sys.argv[1]
    with open(in_file, 'r', encoding='utf-8') as f:
        exprs = [ln.strip() for ln in f if ln.strip()]

    for idx, r in enumerate(exprs, start=1):
        print(f"\n[{idx}] Expresión regular: {r}")
        try:
            nfa = build_nfa(r)
        except Exception as e:
            print(f"  Error al construir AFN: {e}")
            continue

        img_name = f"nfa_{idx}.png"
        try:
            draw_nfa(nfa, img_name)
            print(f"  AFN guardado en {img_name}")
        except Exception as e:
            print(f"  Error al dibujar con Graphviz: {e}")
            continue

        w = input("  Ingrese la cadena w a evaluar: ").strip()
        veredicto = "sí" if nfa.accepts(w) else "no"
        print(f"  w ∈ L(r) ? {veredicto}")

if __name__ == "__main__":
    main()
