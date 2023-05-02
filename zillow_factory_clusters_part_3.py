import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from pykeops.torch import LazyTensor
import time
import torch
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime, date
from geopy.distance import geodesic
import requests
from tqdm import tqdm


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import random



def search_zillow_api(location):
    """Search Zillow API for current listings in the specified location."""
    # API endpoint
    url = "https://zillow56.p.rapidapi.com/search"
    querystring = {"location": location,
                   "rentzestimate": "true",
                   "bedrooms": "1+",
                   "bathrooms": "1+",
                   "squareFeet": "1+",
                   "price": "1+",
                   "latitude": "0+",
                   "livingArea": "0+",
                   "latitude": {"$gt": 0},
                   "lotAreaValue": {"$gt": 100}}
    headers = {
        "X-RapidAPI-Key": "d6492acaa1msh0f4338d49e96062p1d120djsnd28f8e342831",
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)

    # Check if the API call was successful
    if response.status_code == 200:
        print(response.text)
    else:
        print("API call failed with status code:", response.status_code)

    return response

def get_geocode(address):
    """
    Uses Google Maps API to get the geocode of a given address.

    Args:
    address (str): The address to get the geocode for.

    Returns:
    A tuple of latitude and longitude.
    """
    #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     # https: // developers.google.com / maps / billing - and -pricing / pricing  # distance-matrix
    #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     # '        https://www.google.com/maps/embed/v1/MAP_MODE?key=AIzaSyBj2td7I4g-fF9ZV_GqrNALth0_ESoKcTA'
    GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {
        'address': address,
        'sensor': 'false',
        'region': 'texas',
        'key': 'AIzaSyDwz9N4gkiHc8WfqoRMUCERnvXrFexiSAY'
    }

    # Do the request and get the response data
    req = requests.get(GOOGLE_MAPS_API_URL, params=params)
    res = req.json()

    # Use the first result
    try:
        lat_lng = res['results'][0]['geometry']['location']
        return (lat_lng['lat'], lat_lng['lng'])
    except:
        print('Error: could not find geocode for address:', address)
        return None

def get_distances(tx_zillow, tx_Amzn, tesla_sc):
    # Gigafactory Texas
    tgf_geoloc = (30.22, -97.62)

    sc_dist_avg = []
    fc_dist_avg = []
    tgf_dist_km = []
    house_ids = []

    for j in tx_zillow:
        try:
            z_lat = j[5]
            z_long = j[6]
        except:
            print('No Geo data in Zillow datapoint')

        fc_z_dist_km = []
        for i in tx_Amzn:
            fc_z_geodist = geodesic((z_lat, z_long), (tx_Amzn[i][0], tx_Amzn[i][1])).km
            fc_z_dist_km.append(fc_z_geodist)

        sc_z_dist_km = []
        for i in tesla_sc:
            sc_z_geodist = geodesic((z_lat, z_long), (i[8][0], i[8][0])).km
            sc_z_dist_km.append(sc_z_geodist)

        tgf_dist_km.append(geodesic((z_lat, z_long), (tgf_geoloc[0], tgf_geoloc[1])).km)
        sc_dist_avg.append(np.average(sc_z_dist_km))
        fc_dist_avg.append(np.average(fc_z_dist_km))

        house_ids.append(j[0])

    tgf_dist_from_zillow = []
    sc_dist_from_zillow = []
    fc_dist_from_zillow = []

    for i_tfg in tgf_dist_km:
        tgf_dist_from_zillow.append(i_tfg)
    for i_sc in sc_dist_avg:
        sc_dist_from_zillow.append(i_sc)
    for i_fc in fc_dist_avg:
        fc_dist_from_zillow.append(i_fc)

    return tgf_dist_from_zillow, sc_dist_from_zillow, fc_dist_from_zillow

def normalize_dates(date, start_date, scaling_factor):
    """
    Normalize date data based on a specific start date.

    Args:
    dates (list): A list of date strings in the format 'YYYY-MM-DD'.
    start_date (str): The reference start date string in the format 'YYYY-MM-DD'.
    scaling_factor (int): The number of days to normalize the range of values.

    Returns:
    list: A list of normalized date values.
    """

    # for date in dates:
    date_dt = datetime.strptime(date, '%Y-%m-%d').date()
    days_diff = (date_dt - start_date).days
    normalized_date = days_diff / scaling_factor

    return normalized_date

def min_max_scaling(data):
    min_value = min(data)
    max_value = max(data)
    normalized = [(value - min_value) / (max_value - min_value) for value in data]
    return normalized

def z_score_scaling(data):
    mean_value = np.mean(data)
    std_dev_value = np.std(data)
    normalized = [(value - mean_value) / std_dev_value for value in data]
    return normalized

def normalize_data_min_max_scaling(data):
    normalized_data = {}
    for key in data:
        normalized_data[key] = min_max_scaling(data[key])
    return normalized_data

def normalize_data_z_score(data):
    normalized_data = {}
    for key in data:
        normalized_data[key] = z_score_scaling(data[key])
    return normalized_data

def power_iteration(matrix, num_iterations=1000):
    # Initialize a random vector
    vector = torch.randn(matrix.shape[1])

    for _ in tqdm(range(num_iterations), desc="Power Iteration"):
    #for _ in range(num_iterations):
        # Multiply the vector by the matrix
        vector = torch.matmul(matrix, vector)
        # Normalize the vector
        vector = vector / torch.norm(vector)

    # Calculate the corresponding eigenvalue
    eigenvalue = torch.matmul(torch.matmul(matrix, vector), vector) / torch.matmul(vector, vector)

    return eigenvalue, vector

def eigendecomposition(matrix, n_components):
    eigenvalues = []
    eigenvectors = []

    for _ in tqdm(range(n_components), desc="Eigendecomposition"):
    #for _ in range(n_components):
        # Compute the largest eigenvalue and the corresponding eigenvector using power iteration
        eigenvalue, eigenvector = power_iteration(matrix)

        # Subtract the contribution of the current eigenvector from the matrix
        matrix = matrix - eigenvalue * torch.outer(eigenvector, eigenvector)

        # Store the eigenvalue and eigenvector
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

    return torch.tensor(eigenvalues), torch.stack(eigenvectors, dim=1)


def pca_manual(normalized_data, n_components):
    """
    PCA stands for Principal Component Analysis.
     It is a dimensionality reduction technique used in machine learning and statistics
      to transform a high-dimensional dataset into a lower-dimensional space.
       The transformed data retains as much of the original data's variability as possible.
        This is achieved by projecting the data onto the principal components,
         which are the orthogonal axes that capture the most variance in the dataset.
          The technique is often used for visualization, noise reduction, or preprocessing
           before applying other machine learning algorithms.
    """
    # Convert the normalized_data dictionary to a list of lists
    data_list = [normalized_data[key] for key in normalized_data]

    # Transpose the list of lists to have rows as data points and columns as features
    data_list_transposed = list(map(list, zip(*data_list)))

    # Convert the list of lists to a torch tensor
    data_tensor = torch.tensor(data_list_transposed, dtype=torch.float32)

    # Calculate the mean of the data along each feature
    mean = torch.mean(data_tensor, dim=0)

    # Center the data by subtracting the mean
    centered_data = data_tensor - mean

    # Calculate the covariance matrix of the centered data
    covariance_matrix = torch.matmul(centered_data.T, centered_data) / (data_tensor.shape[0] - 1)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    #eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
    # Compute the eigenvectors and eigenvalues of the covariance matrix manually
    eigenvalues, eigenvectors = eigendecomposition(covariance_matrix, n_components)

    # Sort the eigenvectors by decreasing eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the first n_components eigenvectors
    principal_components = sorted_eigenvectors[:, :n_components]

    # Project the data onto the principal components
    reduced_data = torch.matmul(centered_data, principal_components)

    return reduced_data.numpy()

def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in tqdm(range(Niter), desc="K-Means Iterations"):

#    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def reduce_list_length(dictionary, target_length):
    reduced_dict = {}
    for key, value in dictionary.items():
        reduced_dict[key] = random.sample(value, target_length)
    return reduced_dict

def k_distance_sort(data, k):
    n_points = len(data)
    k_distances = []

    for i in range(n_points):
        point = data[i]
        distances = [np.linalg.norm(point - other_point) for other_point in data if not np.array_equal(point, other_point)]
        sorted_distances = sorted(distances)
        k_distances.append(sorted_distances[k - 1])

    k_distances_sorted = sorted(k_distances)
    return k_distances_sorted

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_neighbors(data, point_index, eps):
    neighbors = []
    for i, point in enumerate(data):
        if euclidean_distance(data[point_index], point) <= eps:
            neighbors.append(i)
    return neighbors

def dbscan_manual(data, eps, min_samples):
    labels = np.full(data.shape[0], -1, dtype=int)  # Initialize labels with -1 (unclassified)
    cluster_id = 0
    n_points = len(data)


    #for i in tqdm(range(n_points), desc="DBSCAN Outer Loop"):
    for point_index in range(data.shape[0]):
        if labels[point_index] != -1:  # Skip if the point is already classified
            continue

        neighbors = find_neighbors(data, point_index, eps)

        if len(neighbors) < min_samples:  # Mark as noise if not enough neighbors
            labels[point_index] = -2  # -2 means noise
            continue

        labels[point_index] = cluster_id
        seed_set = set(neighbors)
        seed_set.remove(point_index)

        while seed_set:
            current_point = seed_set.pop()
            if labels[current_point] == -2:  # If the point is noise, assign it to the current cluster
                labels[current_point] = cluster_id

            if labels[current_point] != -1:  # If the point is already assigned, skip it
                continue

            labels[current_point] = cluster_id
            new_neighbors = find_neighbors(data, current_point, eps)
            if len(new_neighbors) >= min_samples:
                seed_set.update(new_neighbors)

        cluster_id += 1

    return labels

def silhouette_score_manual(data, labels):
    n_samples = data.shape[0]
    unique_labels = np.unique(labels)

    # Calculate the average distance between each point and all other points in the same cluster
    a = np.zeros(n_samples)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        for i in label_indices:
            a[i] = np.mean([euclidean_distance(data[i], data[j]) for j in label_indices if j != i])

    # Calculate the average distance between each point and all other points in the nearest cluster
    b = np.zeros(n_samples)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        other_labels = [l for l in unique_labels if l != label]
        for i in label_indices:
            min_other_cluster_mean_distance = float('inf')
            for other_label in other_labels:
                other_label_indices = np.where(labels == other_label)[0]
                mean_distance = np.mean([euclidean_distance(data[i], data[j]) for j in other_label_indices])
                min_other_cluster_mean_distance = min(min_other_cluster_mean_distance, mean_distance)
            b[i] = min_other_cluster_mean_distance

    # Calculate the silhouette score
    silhouette_scores = (b - a) / np.maximum(a, b)
    return np.mean(silhouette_scores)

def calinski_harabasz_score_manual(data, labels):
    n_samples = data.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = unique_labels.shape[0]

    # Calculate the overall mean
    overall_mean = np.mean(data, axis=0)

    # Calculate between-cluster scatter (sum of squared distances between each cluster mean and the overall mean)
    between_cluster_scatter = 0
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_mean = np.mean(cluster_data, axis=0)
        n_cluster_samples = cluster_data.shape[0]
        between_cluster_scatter += n_cluster_samples * np.sum((cluster_mean - overall_mean) ** 2)

    # Calculate within-cluster scatter (sum of squared distances between each point and its cluster mean)
    within_cluster_scatter = 0
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_mean = np.mean(cluster_data, axis=0)
        within_cluster_scatter += np.sum((cluster_data - cluster_mean) ** 2)

    # Calculate Calinski-Harabasz index
    calinski_harabasz_index = ((n_samples - n_clusters) / (n_clusters - 1)) * (between_cluster_scatter / within_cluster_scatter)
    return calinski_harabasz_index

if __name__ == '__main__':

    # Opening JSON file
    # _TX_ZILLOW = "/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/training_data/zillow_texas_json/filtered01.json"

    # _TX_ZILLOW = '/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/atx_house_listings/austinHousingData.csv'
    # _TX_AMZN = "/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/training_data/TX_AMZN_Warehouses.csv"
    # _TESLA_SC = "/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/training_data/texas_tesla_superchargers.csv"
    # _SC_GEO_LOC = "/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/training_data/geo_dict_sc.json"
    # _FC_GEO_LOC = "/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/training_data/geo_dict_fc.json"
    #
    _TX_ZILLOW = "/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/atx_zillow/austinHousingData.csv"
    _TX_AMZN = "/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/self_made/amazon_fullfillment_centers.csv"
    _TESLA_SC = "/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/self_made/texas_tesla_superchargers.csv"
    _FC_GEO_LOC = "/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/self_made/fc_geo_dict.json"

    TX_ZILLOW = []
    with open(_TX_ZILLOW) as file:
        csvreader = csv.reader(file)
        TX_ZILLOW_header = next(csvreader)
        for row in csvreader:
            TX_ZILLOW.append(row)

    TX_AMZN = []
    with open(_TX_AMZN) as file:
        csvreader = csv.reader(file)
        TX_AMZN_header = next(csvreader)
        for row in csvreader:
            TX_AMZN.append(row)

    TESLA_SC = []
    with open(_TESLA_SC) as file:
        csvreader = csv.reader(file)
        TESLA_SC_header = next(csvreader)
        for row in csvreader:
            TESLA_SC.append(row)

    FC_GEO_LOC = []
    with open(_FC_GEO_LOC) as file:
        csvreader = csv.reader(file)
        FC_GEO_LOC_header = next(csvreader)
        for row in csvreader:
            FC_GEO_LOC.append(row)


    # ZILLOW API ############################## Search Zillow API for current listings
    #response = search_zillow_api('austin, texas')
    #z_json = response.json()
    ## Serializing json
    #json_object = json.dumps(z_json, indent=4)
    ## Writing to sample.json
    #with open("training_data/zillow_texas_json/sample01.json", "w") as outfile:
    #   outfile.write(json_object)
    # ///// ZILLOW API ############################## Search Zillow API for current listings

    # Google Maps API ###############################

    #FC_GEO_DICT = dict()  # initialize an empty dictionary
    #for fc in TX_AMZN:
    #   addr = get_geocode(fc[1] + ', ' + fc[2] + ', ' + fc[3] + ', ' + fc[4])
    #FC_GEO_DICT.update({fc[0]: addr})    #out_file = open("fc_geo_dict.json", "w")
    #out_file = open("fc_geo_dict.json", "w")
    #json.dump(FC_GEO_DICT, out_file, indent=4)
    #out_file.close()
#
    #SC_GEO_DICT = dict()  # initialize an empty dictionary
    #for sc in TESLA_SC:
    #   addr = get_geocode(fc[1] + ', ' + fc[2] + ', ' + fc[3] + ', ' + fc[4])
    #SC_GEO_DICT.update({TESLA_SC[sc][0]: (addr)})
    #out_file = open("fc_geo_dict.json", "w")
    #json.dump(SC_GEO_DICT, out_file, indent=4)
    #out_file.close()

    #tgf_dist_from_zillow, sc_dist_from_zillow, fc_dist_from_zillow = get_distances(TX_ZILLOW, FC_GEO_LOC, TESLA_SC)

    # # creating json files
    #out_file = open("tx_sc_dist_from_zillow.json", "w")
    #json.dump(sc_dist_from_zillow, out_file, indent=4)
    #out_file.close()
    #out_file = open("tx_fc_dist_from_zillow.json", "w")
    #json.dump(fc_dist_from_zillow, out_file, indent=4)
    #out_file.close()
    #out_file = open("tx_tgf_dist_from_zillow.json", "w")
    #json.dump(tgf_dist_from_zillow, out_file, indent=4)
    #out_file.close()
    # ///// Google Maps API ##############################

    _tgf_dist_from_z = '/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/self_made/tx_tgf_dist_from_zillow.json'
    _fc_dist_from_z = '/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/self_made/tx_fc_dist_from_zillow.json'
    _sc_dist_from_z = '/Users/joergbln/Desktop/JAH/Code/real_estate_loader/in_Data/self_made/tx_sc_dist_from_zillow.json'

    # _tgf_dist_from_z = '/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/tx_tgf_dist_from_zillow_big.json'
    # _fc_dist_from_z = '/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/tx_fc_dist_from_zillow_big.json'
    # _sc_dist_from_z = '/Users/jordanharris/Code/PycharmProjects/adsense/real_estate_loader/tx_sc_dist_from_zillow_big.json'

    tgf_f = open(_tgf_dist_from_z)
    tgf_dist_from_z = json.load(tgf_f)
    tgf_f.close()

    FC_f = open(_fc_dist_from_z)
    fc_dist_from_z = json.load(FC_f)
    FC_f.close()

    SC_f = open(_sc_dist_from_z)
    sc_dist_from_z = json.load(SC_f)
    SC_f.close()

    # Load housing data
    data = {"tgf": [], "sc": [], "fc": [], "bathrooms": [], "bedrooms": [], "price": [], "sqft": [], "avgSchoolRating": [], "latest_saledate": [], "year_built": []}
    csv_data = {"tgf": [], "sc": [], "fc": [], "bathrooms": [], "bedrooms": [], "price": [], "sqft": [], "avgSchoolRating": [], "latest_saledate": [], "year_built": []}

    # atx_gf_date = date(2020, 7, 22)
    atx_gf_date = datetime.strptime('2020-07-22', '%Y-%m-%d').date()
    scaling_factor = 30

    for zj in TX_ZILLOW:
        data["bedrooms"].append(int(zj[44]))
        data["bathrooms"].append(float(zj[43]))
        data["price"].append(float(zj[18]))
        data["sqft"].append(float(zj[34]))
        data["avgSchoolRating"].append(float(zj[40]))

        date_object = normalize_dates(zj[20], atx_gf_date, scaling_factor)
        data["latest_saledate"].append(date_object)
        year_built_date = datetime.strptime(zj[17], '%Y').date()
        year_object = normalize_dates(str(year_built_date), atx_gf_date, scaling_factor)
        data["year_built"].append(year_object)

        # ###### unnormalized date
        csv_data["bedrooms"].append(int(zj[44]))
        csv_data["bathrooms"].append(float(zj[43]))
        csv_data["price"].append(float(zj[18]))
        csv_data["sqft"].append(float(zj[34]))
        csv_data["avgSchoolRating"].append(float(zj[40]))
        csv_data["latest_saledate"].append(zj[20])
        csv_data["year_built"].append(zj[17])
        # ###### unnormalized date //////

    # json_converter.flatten_list(
    data["tgf"] = tgf_dist_from_z
    data["sc"] = sc_dist_from_z
    data["fc"] = fc_dist_from_z

    # ###### unnormalized date
    csv_data["tgf"] = tgf_dist_from_z
    csv_data["sc"] = sc_dist_from_z
    csv_data["fc"] = fc_dist_from_z
    # ###### unnormalized date //////

    # Calculate the target length
    length = len(data["tgf"])
    target_length = 3000

    # Reduce the length of the lists
    data = reduce_list_length(data, target_length)
    reduced_csv_data = reduce_list_length(csv_data, target_length)

    tgf = torch.tensor(data['tgf'], dtype=torch.float32)
    sc = torch.tensor(data['sc'], dtype=torch.float32)
    fc = torch.tensor(data['fc'], dtype=torch.float32)
    bathrooms = torch.tensor(data['bathrooms'], dtype=torch.float32)
    bedrooms = torch.tensor(data['bedrooms'], dtype=torch.float32)
    price = torch.tensor(data['price'], dtype=torch.float32)
    sqft = torch.tensor(data['sqft'], dtype=torch.float32)
    avgSchoolRating = torch.tensor(data['avgSchoolRating'], dtype=torch.float32)
    latest_saledate = torch.tensor(data['latest_saledate'], dtype=torch.float32)
    year_built = torch.tensor(data['year_built'], dtype=torch.float32)


    ## Load feature stack tensors
    features = torch.stack([tgf, sc, fc, bathrooms, bedrooms, price, sqft, avgSchoolRating, latest_saledate, year_built], dim=1)

    # ###### unnormalized way
    ###### KMEANS
    # Apply KMeans algorithm
    n_clusters = 3
    k_cl, c = KMeans(features, K=n_clusters, Niter=10, verbose=True)

    ###### Calc Scores
    # silhouette_score_k_means = silhouette_score_manual(k_cl.numpy(), features.numpy())
    # calinski_harabasz_score_k_means = calinski_harabasz_score_manual(k_cl.numpy(), features.numpy())
    kmeans_silhouette = silhouette_score(features, k_cl)
    kmeans_calinski_harabasz = calinski_harabasz_score(features, k_cl)
    print("K Means # Silhouette Score: ", kmeans_silhouette)  # labels: array([ 0,  0,  0,  1,  1, -1])
    print("K Means # Calinski-Harabasz Score: ", kmeans_calinski_harabasz)
    ###### Calc Scores ////

    # Separate the data points into different clusters
    clusters = [[] for _ in range(len(c))]
    for i, point in enumerate(features):
        cluster_idx = k_cl[i]
        clusters[cluster_idx].append(point.numpy())

    # Plot the clusters using different colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.stack(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 6], color=colors[i % len(colors)], label=f'Cluster {i}')

    # Plot the cluster centers
    plt.scatter(c[:, 0], c[:, 6], color='black', marker='x', s=200, linewidths=3, label='Cluster centers')

    plt.xlabel("tgf")
    plt.ylabel("sqft")
    plt.legend()
    plt.title('Unnormed KMeans Clustering test (# clusters = {}, (# of DPs = {})'.format(n_clusters, target_length))

    # Add the scores as text annotations
    plt.text(0.05, 0.95, f"Silhouette Score: {kmeans_silhouette:.2f}", transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f"Calinski-Harabasz Score: {kmeans_calinski_harabasz:.2f}", transform=plt.gca().transAxes)

    plt.savefig('unnormed_KMeans Clustering test (n_clusters = {})'.format(n_clusters, target_length))
    plt.show()
    plt.close()
    ###### ///// KMEANS

    # ######  DBSCAN

    features_np = features.numpy()

    n_points = features_np.shape[0]

    # =======
    k = int(np.sqrt(n_points))
    k_d_sort = k_distance_sort(features_np, k)
    # Calculate the average of k-distances
    eps = np.mean(k_d_sort)
    # Define min_samples based on your data dimensions
    min_samples = features_np.shape[1]

    db_cluster_labels = dbscan_manual(features_np, eps, min_samples)
    db_num_clusters = set(db_cluster_labels)

    ###### Calc Scores
    # silhouette_score_dbscan = silhouette_score_manual(db_cluster_labels.numpy(), features_np)
    # calinski_harabasz_score_dbscan = calinski_harabasz_score_manual(db_cluster_labels.numpy(), features_np)
    dbscan_silhouette = silhouette_score(features_np, db_cluster_labels)
    dbscan_calinski_harabasz = calinski_harabasz_score(features_np, db_cluster_labels)
    print("DBSCAN # Silhouette Score: ", dbscan_silhouette)  # labels: array([ 0,  0,  0,  1,  1, -1])
    print("DBSCAN # Calinski-Harabasz Score: ", dbscan_calinski_harabasz)
    ###### Calc Scores ////

    print("Labels: ", db_num_clusters)  # labels: array([ 0,  0,  0,  1,  1, -1])
    # Plot the data points, color-coded by cluster index
    plt.scatter(features[:, 0], features[:, 6], c=db_cluster_labels, cmap='plasma')

    # Add titles, labels
    plt.xlabel("tgf")
    plt.ylabel("sqft")
    plt.legend()

    # Add the scores as text annotations
    plt.text(0.05, 0.95, f"Silhouette Score: {dbscan_silhouette:.2f}", transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f"Calinski-Harabasz Score: {dbscan_calinski_harabasz:.2f}", transform=plt.gca().transAxes)

    plt.title('Unnormed DBSCAN Clustering test (# clusters = {}, (# of DPs = {})'.format(db_num_clusters, target_length))
    plt.savefig('Unnormed_DBSCAN Clustering test (n_clusters = {})'.format(db_num_clusters, target_length))
    plt.show()
    plt.close()

    # # # ======
    new_key = 'cluster'
    reduced_csv_data[new_key] = db_cluster_labels.tolist()
    columns = ["tgf", "sc", "fc", "bathrooms", "bedrooms", "price", "sqft", "avgSchoolRating", "latest_saledate",
               "year_built", "cluster"]
    df = pd.DataFrame(reduced_csv_data, columns=columns)
    df.to_csv("dbscan_date_housing_data_with_clusters.csv", index=False)
    # # # ======

    # ###### ///// DBSCAN
    # ###### ///// unnormalized way



    ###### normalized way
    ###### normalized way
    ###### normalized way
    #!!!!!! Choose the desired normalization method. Comment out the one you don't want to use !!!!!
    # chosen_normalization_method = normalize_data_min_max_scaling
    chosen_normalization_method = normalize_data_z_score

    # Normalize all data points
    normalized_data = chosen_normalization_method(data)

    # Apply PCA to reduce dimensionality (optional)
    pca_data = pca_manual(normalized_data, n_components=2)

    # Convert reduced_data ndarray to torch tensor
    pca_tensor = torch.tensor(pca_data, dtype=torch.float32)

    n_clusters = 3

    # Apply KMeans algorithm
    n_k_cl, n_c = KMeans(pca_tensor, K=n_clusters, Niter=10, verbose=True)

    ###### Calc Scores
    # n_silhouette_score_k_means = silhouette_score_manual(n_k_cl.numpy(), pca_tensor.numpy())
    # n_calinski_harabasz_score_k_means = calinski_harabasz_score_manual(n_k_cl.numpy(), pca_tensor.numpy())
    n_kmeans_silhouette = silhouette_score(pca_tensor, n_k_cl)
    n_kmeans_calinski_harabasz = calinski_harabasz_score(pca_tensor, n_k_cl)
    print("K Means # Silhouette Score: ", n_kmeans_silhouette)  # labels: array([ 0,  0,  0,  1,  1, -1])
    print("K Means # Calinski-Harabasz Score: ", n_kmeans_calinski_harabasz)  # labels: array([ 0,  0,  0,  1,  1, -1])
    ###### Calc Scores ////


    # Separate the data points into different clusters
    clusters = [[] for _ in range(len(n_c))]
    for i, point in enumerate(pca_tensor):
        cluster_idx = n_k_cl[i]
        clusters[cluster_idx].append(point.numpy())

    # Plot the clusters using different colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.stack(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % len(colors)], label=f'Cluster {i}')

    # Plot the cluster centers
    plt.scatter(n_c[:, 0], n_c[:, 1], color='black', marker='x', s=200, linewidths=3, label='Cluster centers')

    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.legend()

    # Add the scores as text annotations
    plt.text(0.05, 0.95, f"Norm Silhouette Score: {n_kmeans_silhouette:.2f}", transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f"Norm Calinski-Harabasz Score: {n_kmeans_calinski_harabasz:.2f}", transform=plt.gca().transAxes)

    plt.title('Normed KMeans Clustering test (# clusters = {}, (# of DPs = {})'.format(n_clusters, target_length))
    plt.savefig('normed_KMeans Clustering test (n_clusters = {})'.format(n_clusters, target_length))
    plt.show()
    plt.close()
    #
    # # # # ======
    # # # Create a pandas DataFrame with the updated data for graphing
    columns = ["tgf", "sc", "fc", "bathrooms", "bedrooms", "price", "sqft", "avgSchoolRating", "latest_saledate", "year_built", "cluster"]
    new_key = 'cluster'
    reduced_csv_data[new_key] = k_cl.tolist()
    df = pd.DataFrame(reduced_csv_data, columns=columns)
    df.to_csv("kmeans_date_housing_data_with_normed_clusters.csv", index=False)
    # # ======

    # ###### ///// DBSCAN
    n_points = pca_data.shape[0]

    # =======
    k = int(np.sqrt(n_points))
    k_d_sort = k_distance_sort(pca_data, k)
    # Calculate the average of k-distances
    eps = np.mean(k_d_sort)
    # Define min_samples based on your data dimensions
    min_samples = pca_data.shape[1]

    db_cluster_labels_norm = dbscan_manual(pca_data, eps, min_samples)
    db_num_clusters_norm = set(db_cluster_labels_norm)
    print("Labels: ", set(db_num_clusters_norm))  # labels: array([ 0,  0,  0,  1,  1, -1])

    ###### Calc Scores
    # n_silhouette_score_dbscan = silhouette_score_manual(db_cluster_labels_norm.numpy(), pca_data.numpy())
    # n_calinski_harabasz_score_dbscan = calinski_harabasz_score_manual(db_cluster_labels_norm.numpy(), pca_data.numpy())
    n_dbscan_silhouette = silhouette_score(pca_data, db_cluster_labels_norm)
    n_dbscan_calinski_harabasz = calinski_harabasz_score(pca_data, db_cluster_labels_norm)
    print("DBSCAN # Silhouette Score: ", n_dbscan_silhouette)  # labels: array([ 0,  0,  0,  1,  1, -1])
    print("DBSCAN # Calinski-Harabasz Score: ", n_dbscan_calinski_harabasz)
    ###### Scores

    # Plot the data points, color-coded by cluster index
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=db_cluster_labels_norm, cmap='plasma')

    # Add titles, labels
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.legend()
    plt.title('Normed DBSCAN Clustering test (# clusters = {}, (# of DPs = {})'.format(db_num_clusters_norm, target_length))

    # Add the scores as text annotations
    plt.text(0.05, 0.95, f"Silhouette Score: {n_dbscan_silhouette:.2f}", transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f"Calinski-Harabasz Score: {n_dbscan_calinski_harabasz:.2f}", transform=plt.gca().transAxes)

    plt.savefig('normed_DBSCAN Clustering test (n_clusters = {})'.format(db_num_clusters_norm, target_length))
    plt.show()
    plt.close()

    # # # # ======
    new_key = 'cluster'
    reduced_csv_data[new_key] = db_cluster_labels_norm.tolist()
    columns = ["tgf", "sc", "fc", "bathrooms", "bedrooms", "price", "sqft", "avgSchoolRating", "latest_saledate",
               "year_built", "cluster"]
    df = pd.DataFrame(reduced_csv_data, columns=columns)
    df.to_csv("norm_dbscan_date_housing_data_with_clusters.csv", index=False)
    # # ======
    # ###### ///// DBSCAN




