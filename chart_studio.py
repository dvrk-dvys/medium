# https://jennifer-banks8585.medium.com/how-to-embed-interactive-plotly-visualizations-on-medium-blogs-710209f93bd#:~:text=Visit%20your%20Plotly%20account%2C%20go,over%20it%20and%20click%20Viewer.&text=Now%20copy%20the%20url%20and%20paste%20it%20in%20your%20Medium%20blog.
import chart_studio
import csv

chart_studio.tools.set_credentials_file(username='dvrk-dvys', api_key='Kz7DoATTSKRSHMeX8vZw')

# username='your_username'
# api_key='your_api_key'
# chart_studio.tools.set_credentials_file(username=username,
#                                         api_key=api_key)

_DBSCAN = "unnormed_dbscan_housing_data_with_clusters.csv"
DBSCAN = []
cluster_labels = []
with open(_DBSCAN) as file:
    csvreader = csv.reader(file)
    DBSCAN_header = next(csvreader)
    for row in csvreader:
        DBSCAN.append(row)
        cluster_labels.append(row[10])

# Find the number of unique clusters
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

# Separate the data points into different clusters
clusters = [[] for _ in range(n_clusters)]
noise = []

for i, d in enumerate(DBSCAN):
    cluster_idx = cluster_labels[i]
    if cluster_idx == -1:
        noise.append(d[0])
    else:
        cluster_assignment = int(float(d[-1]))
        clusters[cluster_assignment].append(d)




df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length",
                     color="species", size='petal_length',
                     hover_data=['petal_width']
                  )

py.plot(fig, filename="plotly_scatter", auto_open = True)
fig.show()