import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import folium
import networkx as nx
import matplotlib.pyplot as plt
import os

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function Definitions
def calculate_distance_matrix(coordinates):
    return pdist(coordinates)

def mst_solver(distance_matrix):
    mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)
    return mst

def two_opt(route, distance_matrix):
    best_route = route
    best_distance = total_distance(best_route, distance_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                new_distance = total_distance(new_route, distance_matrix)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        route = best_route
    return best_route

def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

def optimize_routes_for_cluster(cluster_data, depot_coords):
    coords = cluster_data[['Latitude', 'Longitude']].values
    all_coords = np.vstack([depot_coords, coords])
    distance_matrix = squareform(calculate_distance_matrix(all_coords))
    mst_matrix = mst_solver(distance_matrix)

    # Get MST edges and create initial route based on MST
    mst_edges = np.transpose(np.nonzero(mst_matrix))
    initial_route = [0]
    for i, j in mst_edges:
        if j not in initial_route:
            initial_route.append(j)
        if len(initial_route) == len(all_coords):
            break
    initial_route.append(0)  # Return to depot

    optimized_route = two_opt(initial_route, distance_matrix)
    return optimized_route, mst_matrix

# Set environment variable to avoid memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load dataset from Excel file
employee_data = pd.read_excel('/content/Kalyani_Bus_Stop_lat_long_copy1 (1).xlsx')

# Define depot coordinates
depot_coords = [18.791448, 74.293129]

# Cluster employees into groups of 30
n_employees = employee_data.shape[0]
kmeans = KMeans(n_clusters=int(np.ceil(n_employees / 30)), random_state=0, n_init='auto').fit(employee_data[['Latitude', 'Longitude']])
employee_data['Cluster'] = kmeans.labels_
clusters = employee_data.groupby('Cluster')

# Verify that all employees are assigned to a cluster
assert employee_data['Cluster'].isnull().sum() == 0, "Some employees are not assigned to a cluster."

# Optimize routes for each cluster
routes = {}
msts = {}
for cluster_id, cluster_data in clusters:
    routes[cluster_id], mst_matrix = optimize_routes_for_cluster(cluster_data, depot_coords)
    msts[cluster_id] = mst_matrix

# Plot the routes on a Folium map
map_center = depot_coords
route_map = folium.Map(location=map_center, zoom_start=12)

# Add depot marker
folium.Marker(location=depot_coords, icon=folium.Icon(color='red'), popup='Depot').add_to(route_map)

# Function to add route to map
def add_route_to_map(route, cluster_data, map_obj, color, depot_coords):
    coords = cluster_data[['Latitude', 'Longitude']].values
    all_coords = np.vstack([depot_coords, coords])
    route_coords = [all_coords[i] for i in route]
    folium.PolyLine(route_coords, color=color, weight=2.5, opacity=1).add_to(map_obj)
    for point in route_coords:
        folium.Marker(location=point, icon=folium.Icon(color='blue'), popup=str(point)).add_to(map_obj)

# Colors for different clusters
colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred',  'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

for cluster_id, route in routes.items():
    cluster_data = clusters.get_group(cluster_id)
    add_route_to_map(route, cluster_data, route_map, colors[cluster_id % len(colors)], depot_coords)

route_map.save('optimized_routes.html')
route_map

# Display MST
def display_mst(mst_matrix, coordinates):
    G = nx.Graph()
    for i in range(len(coordinates)):
        G.add_node(i, pos=coordinates[i])

    mst_edges = np.transpose(np.nonzero(mst_matrix))
    for i, j in mst_edges:
        if mst_matrix[i, j] > 0:
            G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    plt.title('Minimum Spanning Tree (MST)')
    plt.show()

# Display MST for each cluster
for cluster_id, mst_matrix in msts.items():
    cluster_data = clusters.get_group(cluster_id)
    coords = cluster_data[['Latitude', 'Longitude']].values
    all_coords = np.vstack([depot_coords, coords])
    display_mst(mst_matrix, all_coords)