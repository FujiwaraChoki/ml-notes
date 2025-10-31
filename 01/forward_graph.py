from micrograd.engine import Value
from graphviz import Digraph

# ---- Forward pass ----
a = Value(-4.0)
b = Value(2.0)

c = a + b
d = a * b + b**3
c = c + c + 1
c = c + 1 + c + (-a)
d = d + d * 2 + (b + a)  # remove .relu()
d = d + 3 * d + (b - a)  # remove .relu()
e = c - d
f = e**2
g = f / 2.0
g = g + 10.0 / f
print(f"Forward result g = {g.data:.4f}")

# ---- Backward pass ----
g.backward()
print(f"Gradient of a: {a.grad:.4f}")
print(f"Gradient of b: {b.grad:.4f}")

# ---- Graph builder ----
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        label = f"data={n.data:.2f}, grad={n.grad:.2f}"
        dot.node(name=uid, label=label, shape='record')
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))
    return dot

dot = draw_dot(g)
try:
    dot.render('forward_pass_graph', view=True)
    print("Graph saved to forward_pass_graph.png")
except Exception as e:
    print(f"\nWarning: Could not render graph: {e}")
    print("To fix this, install Graphviz:")
    print("  macOS: brew install graphviz")
    print("  Ubuntu/Debian: sudo apt-get install graphviz")
    print("  Windows: choco install graphviz")
    # Save the DOT source code anyway
    with open('forward_pass_graph.dot', 'w') as f:
        f.write(dot.source)
    print("DOT source saved to forward_pass_graph.dot")
