from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.geo import haversine_distance

seed = 99
np.random.seed(seed)

n_simulations = 10
n_nodes = 10
k= 3

cell_dataset = pd.read_parquet("assets/porto_cells.parquet")
output = f"porto_{n_nodes}n_{k}k"
bbox_img = "assets/BBox_Porto.png"
bbox_boundaries = "assets/BBox_Porto.txt"

def build_network(
    cell_dataset: pd.DataFrame, num_towers: int, k_edge_connectivity: int
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Builds a network of towers.

    :param cell_dataset: the dataset containing tower positions
    :param num_towers: the number of towers (nodes) to use for the network
    :param k_edge_connectivity: k edge connectivity of the desired graph
    :return: a tuple containing as first element the networkx graph object, as second element a
    dataframe with node coordinates
    """
    while True:  # when a result is found, it is directly returned
        # sample the required number of towers
        towers = cell_dataset.sample(num_towers, replace=False)
        towers.reset_index(drop=True, inplace=True)

        # calculate tuples of node distances (v, w, dist(v,w))
        distances = set()
        for i in range(num_towers):
            for j in range(i + 1, num_towers):
                dist = haversine_distance(
                    longitude_1=towers["lon"].astype(float)[i],
                    longitude_2=towers["lon"].astype(float)[j],
                    latitude_1=towers["lat"].astype(float)[i],
                    latitude_2=towers["lat"].astype(float)[j],
                )
                distances.add((i, j, dist))

        network = nx.Graph()
        network.add_nodes_from(range(num_towers))

        edges = nx.k_edge_augmentation(network, k_edge_connectivity, avail=distances)
        network.add_edges_from(edges)

        return network, towers[["lon", "lat"]]

if __name__ == "__main__":
    for i in range(n_simulations):
        folder_path = Path(f"data/networks/{output}/{i}")
        folder_path.mkdir(parents=True, exist_ok=True)

        network, towers = build_network(cell_dataset, num_towers=n_nodes, k_edge_connectivity=k)

        nx.write_adjlist(network, folder_path / "adj_list.txt")
        towers.to_csv(folder_path / "towers.csv", index=False)

        print(f"Saved network {i}!")
