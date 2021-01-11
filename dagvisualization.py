import networkx as nx
from preprocess.XMLProcess import XML2DAG


graph = nx.DiGraph()
wfname = './datasets/CyberShake_30.xml'
WF = XML2DAG(wfname, 50)
WF.get_dag()
edges = WF.print_graph()
graph.add_edges_from(edges)

print(graph.nodes()) # => NodeView(('root', 'a', 'b', 'e', 'c', 'd'))
# nx.shortest_path(graph, 'root', 'e') # => ['root', 'a', 'e']
# nx.dag_longest_path(graph) # => ['root', 'a', 'b', 'd', 'e']
# list(nx.topological_sort(graph)) # => ['root', 'a', 'b', 'd', 'e', 'c']
print(list(nx.topological_sort(graph)))
print(nx.is_directed(graph)) # => True
print(nx.is_directed_acyclic_graph(graph)) # => True

from matplotlib import pyplot as plt
g1 = nx.DiGraph()
g1.add_edges_from(edges)
plt.tight_layout()
nx.draw_networkx(g1, arrows=True)
plt.savefig("g1.png", format="PNG")
# tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
plt.clf()