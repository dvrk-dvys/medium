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
        "X-RapidAPI-Key": "YOUR API KEY",
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)

    # Check if the API call was successful
    if response.status_code == 200:
        print(response.text)
    else:
        print("API call failed with status code:", response.status_code)

    return response

def load_and_filter_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    filtered_data = []
    required_keys = ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude']

    for obj in data['results']:
        if all(key in obj for key in required_keys) and ('lotAreaUnit' in obj or 'livingArea' in obj):
            if all(int(obj[val]) != 0 for val in required_keys) and (obj not in filtered_data):
                filtered_data.append(obj)

    # Serializing json
    json_object = json.dumps(filtered_data, indent=4)
    # Writing to sample.json
    with open("filtered01.json", "w") as outfile:
        outfile.write(json_object)

    return filtered_data

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

    return cl,

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Opening JSON file
    # _TX_ZILLOW = "/in_Data/zillow_texas_json/sample01.json"
    _TX_ZILLOW = "/in_Data/atx_zillow/austinHousingData.csv"
    _TX_AMZN = "/in_Data/amazon_fullfillment_centers.csv"
    _TESLA_SC = "/in_Data/texas_tesla_superchargers.csv"
    _FC_GEO_LOC = "/Users/joergbln/Desktop/JAH/Code/in_Data/self_made/fc_geo_dict.json"


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
    response = search_zillow_api('austin, texas')
    z_json = response.json()
    # Serializing json
    json_object = json.dumps(z_json, indent=4)
    # Writing to sample.json
    with open("/in_Data/zillow_texas_json/sample01.json", "w") as outfile:
        outfile.write(json_object)
    # //// ZILLOW API ##############################

    # Google Maps API ###############################
    FC_GEO_DICT = dict()  # initialize an empty dictionary
    for fc in TX_AMZN:
        addr = get_geocode(fc[1] + ', ' + fc[2] + ', ' + fc[3] + ', ' + fc[4])
    FC_GEO_DICT.update({fc[0]: addr})  # out_file = open("fc_geo_dict.json", "w")
    out_file = open("fc_geo_dict.json", "w")
    json.dump(FC_GEO_DICT, out_file, indent=4)
    out_file.close()

    SC_GEO_DICT = dict()  # initialize an empty dictionary
    for sc in TESLA_SC:
        addr = get_geocode(fc[1] + ', ' + fc[2] + ', ' + fc[3] + ', ' + fc[4])
    SC_GEO_DICT.update({TESLA_SC[sc][0]: (addr)})
    out_file = open("fc_geo_dict.json", "w")
    json.dump(SC_GEO_DICT, out_file, indent=4)
    out_file.close()

    tgf_dist_from_zillow, sc_dist_from_zillow, fc_dist_from_zillow = get_distances(TX_ZILLOW, FC_GEO_LOC, TESLA_SC)

    # creating json files
    out_file = open("tx_sc_dist_from_zillow.json", "w")
    json.dump(sc_dist_from_zillow, out_file, indent=4)
    out_file.close()
    out_file = open("tx_fc_dist_from_zillow.json", "w")
    json.dump(fc_dist_from_zillow, out_file, indent=4)
    out_file.close()
    out_file = open("tx_tgf_dist_from_zillow.json", "w")
    json.dump(tgf_dist_from_zillow, out_file, indent=4)
    out_file.close()
    # ///// Google Maps API ##############################

    _tgf_dist_from_z = '/Users/joergbln/Desktop/JAH/Code/in_Data/self_made/tx_tgf_dist_from_zillow.json'
    _fc_dist_from_z = '/Users/joergbln/Desktop/JAH/Code/in_Data/self_made/tx_fc_dist_from_zillow.json'
    _sc_dist_from_z = '/Users/joergbln/Desktop/JAH/Code/in_Data/self_made/tx_sc_dist_from_zillow.json'

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
    data = {"tgf": [], "sc": [], "fc": [], "bathrooms": [], "bedrooms": [], "price": [], "sqft": [],
            "avgSchoolRating": [], "latest_saledate": [], "year_built": []}

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

    # json_converter.flatten_list(
    data["tgf"] = tgf_dist_from_z
    data["sc"] = sc_dist_from_z
    data["fc"] = fc_dist_from_z

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

    n_clusters = 3
    cl, c = KMeans(features, K=n_clusters, Niter=10, verbose=True)

    # Separate the data points into different clusters
    clusters = [[] for _ in range(len(c))]
    for i, point in enumerate(features):
        cluster_idx = cl[i]
        clusters[cluster_idx].append(point.numpy())
    # Plot the clusters using different colors

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.stack(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 9], color=colors[i % len(colors)], label=f'Cluster {i}')

    # Plot the cluster centers
    plt.scatter(c[:, 0], c[:, 9], color='black', marker='x', s=200, linewidths=3, label='Cluster centers')

    plt.xlabel('Distance to TGF')
    plt.ylabel('Year Built')
    plt.legend()
    plt.title('KMeans Clustering test (# clusters = ' + str(n_clusters) + ')' + ' (# of DPs = ' + str(target_length) + ')')
    plt.savefig('KMeans Clustering test (# clusters = ' + str(n_clusters) + ')' + ' (# of DPs = ' + str(target_length) + ')')
    plt.show()
    plt.close()

    # Plot the clusters using different colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.stack(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 5], color=colors[i % len(colors)], label=f'Cluster {i}')

    # Plot the cluster centers
    plt.scatter(c[:, 0], c[:, 5], color='black', marker='x', s=200, linewidths=3, label='Cluster centers')

    plt.xlabel('Distance to TGF')
    plt.ylabel('Price')
    plt.legend()
    plt.title('KMeans Clustering test (# clusters = ' + str(n_clusters) + ')' + ' (# of DPs = ' + str(target_length) + ')')
    plt.savefig('tgf_latest_saledate_KMeans Clustering test (# clusters = ' + str(n_clusters) + ')' + ' (# of DPs = ' + str(target_length) + ')')
    plt.show()
    plt.close()

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the clusters using different colors
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.stack(cluster)
            ax.scatter(cluster[:, 0], cluster[:, 4], cluster[:, 5], color=colors[i % len(colors)], label=f'Cluster {i}')

    # Plot the cluster centers
    ax.scatter(c[:, 0], c[:, 4], c[:, 5], color='black', marker='x', s=200, linewidths=3, label='Cluster centers')

    ax.set_xlabel('Distance to TGF')
    ax.set_ylabel('Number of Bedrooms')
    ax.set_zlabel('Price')
    ax.legend()

    ax.view_init(30, 45)  # You can adjust these values to change the viewing angle

    plt.title('KMeans Clustering 3D test (n_clusters = {})'.format(n_clusters))
    plt.savefig('KMeans Clustering 3D test (n_clusters = {})'.format(n_clusters))
    plt.show()
    plt.close()

    # Convert the features tensor to a NumPy array
    features_np = features.numpy()

    # Add the cluster labels as a new column
    data_with_labels = np.column_stack((features_np, cl.numpy()))

    # Create a pandas DataFrame with the updated data
    columns = ["tgf", "sc", "fc", "bathrooms", "bedrooms", "price", "sqft", "avgSchoolRating", "latest_saledate",
               "year_built", "cluster"]
    df = pd.DataFrame(data_with_labels, columns=columns)

    # Save the DataFrame as a CSV file
    df.to_csv("housing_data_with_clusters.csv", index=False)





