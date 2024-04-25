import matplotlib.pyplot as plt
import networkx as nx

# Define the nodes and their labels
nodes = ['Load Audio File', 'Extract MFCC Features', 'Compute Mean and Covariance',
         'Create Feature Vector', 'Get Nearest Neighbors', 'Predict Genre', 'Display Result']

# Define the edges and their labels
edges = [('Load Audio File', 'Extract MFCC Features', 'Audio Signal'),
         ('Extract MFCC Features', 'Compute Mean and Covariance', 'MFCC Features'),
         ('Compute Mean and Covariance', 'Create Feature Vector', 'Mean and Covariance'),
         ('Create Feature Vector', 'Get Nearest Neighbors', 'Feature Vector'),
         ('Get Nearest Neighbors', 'Predict Genre', 'Nearest Neighbors'),
         ('Predict Genre', 'Display Result', 'Predicted Genre')]

# Create the graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from([(u, v, {'label': l}) for u, v, l in edges])

# Set the layout and draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): l for u, v, l in edges}, font_size=8)
plt.title('Block Diagram of Music Genre Prediction System')
plt.show()