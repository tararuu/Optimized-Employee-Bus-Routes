import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import folium
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function Definitions
def calculate_distance_matrix(coordinates):
    return squareform(pdist(coordinates))

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
    distance_matrix = calculate_distance_matrix(all_coords)
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
    return optimized_route

def tlbo_clustering(employee_data, num_clusters, max_iter=100):

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(employee_data[['Latitude', 'Longitude']])
    employee_data['Cluster'] = kmeans.labels_

    population = [kmeans.cluster_centers_ for _ in range(10)]
    best_solution = min(population, key=lambda x: kmeans.inertia_)

    for _ in range(max_iter):
        # Teacher Phase
        teacher = min(population, key=lambda x: kmeans.inertia_)
        for i in range(len(population)):
            new_solution = teacher + np.random.rand(*teacher.shape) * (teacher - np.mean(population, axis=0))
            new_solution_kmeans = KMeans(n_clusters=num_clusters, init=new_solution, n_init=1, random_state=0).fit(employee_data[['Latitude', 'Longitude']])
            if new_solution_kmeans.inertia_ < kmeans.inertia_:
                population[i] = new_solution_kmeans.cluster_centers_

        # Learner Phase
        for i in range(len(population)):
            partner_index = np.random.choice([x for x in range(len(population)) if x != i])
            if kmeans.inertia_ > KMeans(n_clusters=num_clusters, init=population[partner_index], n_init=1, random_state=0).fit(employee_data[['Latitude', 'Longitude']]).inertia_:
                population[i] = population[partner_index]

        best_solution = min(population, key=lambda x: KMeans(n_clusters=num_clusters, init=x, n_init=1, random_state=0).fit(employee_data[['Latitude', 'Longitude']]).inertia_)

    final_kmeans = KMeans(n_clusters=num_clusters, init=best_solution, n_init=1, random_state=0).fit(employee_data[['Latitude', 'Longitude']])
    employee_data['Cluster'] = final_kmeans.labels_
    return employee_data

def save_routes_to_txt(routes, clusters, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for cluster_id, route in routes.items():
        cluster_data = clusters.get_group(cluster_id)
        file_path = os.path.join(directory, f"Route_{cluster_id}.txt")
        with open(file_path, 'w') as f:
            f.write(f"Route for Cluster {cluster_id}:\n")
            f.write(f"Start  {depot_coords[0]}  {depot_coords[1]}\n")
            for i, node in enumerate(route):
                if node == 0:  # Depot
                    continue
                node_idx = node - 1  # Adjust for 0-indexing
                if node_idx < len(cluster_data):
                    f.write(f"N{i}  {cluster_data.iloc[node_idx]['Latitude']}  {cluster_data.iloc[node_idx]['Longitude']}\n")
            f.write(f"End  {depot_coords[0]}  {depot_coords[1]}\n")

# Set environment variable to avoid memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load dataset from Excel file
file_path = r'/content/Kalyani_Bus_Stop_lat_long_copy1 (1).xlsx'
try:
    employee_data = pd.read_excel(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file at {file_path} was not found. Please check the file path and ensure it is correct.")

# Define depot coordinates
depot_coords = [18.791448, 74.293129]

# Cluster employees into groups of 30 using TLBO
n_employees = employee_data.shape[0]
num_clusters = int(np.ceil(n_employees / 33))
employee_data = tlbo_clustering(employee_data, num_clusters)
clusters = employee_data.groupby('Cluster')

# Verify that all employees are assigned to a cluster
assert employee_data['Cluster'].isnull().sum() == 0, "Some employees are not assigned to a cluster."

# Optimize routes for each cluster
routes = {}
for cluster_id, cluster_data in clusters:
    routes[cluster_id] = optimize_routes_for_cluster(cluster_data, depot_coords)

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

save_routes_to_txt(routes, clusters, 'routes1')

# Save the map as an HTML file
route_map.save('optimized_routes.html')
route_map
