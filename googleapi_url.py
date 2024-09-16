

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
import os
import warnings
import googlemaps

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function Definitions
def get_distance_matrix(coords, api_key):
    """
    Get the pairwise distance matrix using the Google Maps Distance Matrix API.

    Returns:
    array: Pairwise distance matrix.
    """
    gmaps = googlemaps.Client(key=api_key)
    n = len(coords)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                origins = [coords[i]]
                destinations = [coords[j]]
                result = gmaps.distance_matrix(origins, destinations, mode='driving')
                distance = result['rows'][0]['elements'][0]['distance']['value']
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

    return distance_matrix#array

def mst_solver(distance_matrix):
    """
    Solve the Minimum Spanning Tree (MST) for the given distance matrix.

    Returns:
    array: MST represented as a sparse matrix.
    """
    mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)#imported mst lib
    return mst

def two_opt(route, distance_matrix):
    """
    Optimize the given route using the 2-opt algorithm.

    Returns:
    list: Optimized route.
    """
    best_route = route#list
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
    """
    Calculate the total distance of the given route.

    Returns:
    float: Total distance of the route.
    """
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

def optimize_routes_for_cluster(cluster_data, depot_coords, api_key):
    """
    Optimize the routes for a given cluster using MST and 2-opt algorithms.

    Returns:
    list: Optimized route.
    """
    coords = cluster_data[['Latitude', 'Longitude']].values#Data of the cluster.
    all_coords = np.vstack([depot_coords, coords])
    distance_matrix = get_distance_matrix(all_coords, api_key)
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
    """
    Perform clustering using TLBO algorithm.

    Returns:
    DataFrame: Data with cluster assignments.
    """
    #num_clusters: Number of clusters.
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(employee_data[['Latitude', 'Longitude']])
    employee_data['Cluster'] = kmeans.labels_#emp data get data of empw ith lat long

    population = [kmeans.cluster_centers_ for _ in range(10)]
    best_solution = min(population, key=lambda x: kmeans.inertia_)

    for _ in range(max_iter):#iteration defination for tlbo algo
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

def generate_google_maps_url(route, cluster_data, depot_coords):
    """
    Generate a Google Maps URL to visualize the route.

    Returns:
    str: Google Maps URL.
    """
    coords = cluster_data[['Latitude', 'Longitude']].values
    all_coords = [depot_coords] + [list(coords[i]) for i in range(len(coords))]
    route_coords = [all_coords[i] for i in route]
    waypoints = '/'.join([f'{lat},{lon}' for lat, lon in route_coords[1:-1]])
    url = f"https://www.google.com/maps/dir/{depot_coords[0]},{depot_coords[1]}/{waypoints}/{depot_coords[0]},{depot_coords[1]}"
    return url

# Set environment variable to avoid memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load dataset from Excel file
file_path = r'/content/Kalyani_Bus_Stop_lat_long_copy.xlsx'
try:
    employee_data = pd.read_excel(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file at {file_path} was not found. Please check the file path and ensure it is correct.")

# Define depot coordinates
depot_coords = [18.791448, 74.293129]

# Google Maps API key
api_key = 'YOUR_OWN_GOOGLE MAPS API KEY WITH DISTANCE MATRIX ENABLED'  # Replace with your actual API key

# Cluster employees into groups of 30 using TLBO
n_employees = employee_data.shape[0]
num_clusters = int(np.ceil(n_employees / 30))
employee_data = tlbo_clustering(employee_data, num_clusters)
clusters = employee_data.groupby('Cluster')

# Verify that all employees are assigned to a cluster
assert employee_data['Cluster'].isnull().sum() == 0, "Some employees are not assigned to a cluster."

# Optimize routes for each cluster and generate Google Maps URLs
routes = {}
google_maps_urls = []
for cluster_id, cluster_data in clusters:
    optimized_route = optimize_routes_for_cluster(cluster_data, depot_coords, api_key)
    routes[cluster_id] = optimized_route
    google_maps_url = generate_google_maps_url(optimized_route, cluster_data, depot_coords)
    google_maps_urls.append(google_maps_url)
    print(f"Cluster {cluster_id} Route URL: {google_maps_url}")
