import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score as ss

def minimums(values_array: np.array) -> np.array:
    
    # Retrieve indices of smallest values across the second dimension.
    minimum_indices = np.argmin(values_array, axis=1)

    # (n-points,) matrix of smallest values.
    return values_array[np.arange(values_array.shape[0]), minimum_indices]

def points_to_points_distances(points_a: np.array, points_b: np.array) -> np.array:

    # Reshape to enable broadcasting.
    # Results in a (n, 1, d) matrix, s.t.:
    # n <- number of points in points_a,
    # d <- number of dimensions in both points_a and points_b.
    # Middle 1 dimension allows broadcasting across all points.
    points_expanded = points_a[:, np.newaxis, :]

    # Differences of dimensions across all pairs of points.
    # Results in a (n, m, d) matrix, s.t.:
    # n <- number of points in points_a,
    # m <- number of points in points_b,
    # d <- number of dimensions in both points_a and points_b.
    differences = points_expanded - points_b

    differences_squared = differences**2

    # Sum the squared differences of the dimensions -- along the final axis.
    # Results in a (n, m) matrix, s.t.:
    # n <- number of points in points_a,
    # m <- number of points in points_b.
    sum_of_squared_differences = np.sum(differences_squared, axis=2)
    
    return sum_of_squared_differences


def single_point_in_densest_area(points_a: np.array, points_b: np.array, max_block_size: int) -> tuple[float, float]:

    # Results in the point nearest to all other points.

    distance_sums = single_point_in_densest_area_i(points_a=points_a, points_b=points_b, distance_sums=np.array([]), i=0, max_block_size=max_block_size)

    return points_a[np.argmin(distance_sums)]

def single_point_in_densest_area_i(points_a: np.array, points_b: np.array, distance_sums: np.array, i: int, max_block_size: int) -> np.array:

    if i >= points_a.shape[0]:
        return distance_sums.reshape(1, points_a.shape[0])
    
    block_size = min(points_a.shape[0] - i, max_block_size)

    points_a_block = points_a[i:i+block_size]

    distance_sums_block = np.zeros(block_size)

    new_distance_sums_block = single_point_in_densest_area_j(points_a_block=points_a_block, points_b=points_b, distance_sums_block=distance_sums_block, j=0, max_block_size=max_block_size)

    append_distance_sums = np.append(distance_sums, new_distance_sums_block)

    return single_point_in_densest_area_i(points_a=points_a, points_b=points_b, distance_sums=append_distance_sums, i=i+block_size, max_block_size=max_block_size)

def single_point_in_densest_area_j(points_a_block: np.array, points_b: np.array, distance_sums_block: np.array, j: int, max_block_size: int) -> np.array:

    if j >= points_b.shape[0]:
        return distance_sums_block
    
    block_size = min(points_b.shape[0] - j, max_block_size)

    points_b_block = points_b[j:j+block_size]

    new_distance_sums_block = distance_sums_block + np.sum(points_to_points_distances(points_a=points_a_block, points_b=points_b_block), axis=1)
    
    return single_point_in_densest_area_j(points_a_block=points_a_block, points_b=points_b, distance_sums_block=new_distance_sums_block, j=j+block_size, max_block_size=max_block_size)


def distances_to_nearest_centroid(points: np.array, centroids: np.array, max_block_size: int) -> np.array:

    return distances_to_nearest_centroid_i(points=points, centroids=centroids, minimum_distances=np.array([]), i=0, max_block_size=max_block_size)

def distances_to_nearest_centroid_i(points: np.array, centroids: np.array, minimum_distances: np.array, i: int, max_block_size: int) -> np.array:

    if i >= points.shape[0]:
        return minimum_distances.reshape(1, points.shape[0])
    
    block_size = min(points.shape[0] - i, max_block_size)

    points_block  = points[i:i+block_size]

    minimum_distances_block = np.full((block_size), fill_value=np.inf)
    
    new_minimum_distances_block = distances_to_nearest_centroid_j(points_block=points_block, centroids=centroids, minimum_distances_block=minimum_distances_block, j=0, max_block_size=max_block_size)
    
    append_minimum_distances_block = np.append(minimum_distances, new_minimum_distances_block)

    return distances_to_nearest_centroid_i(points=points, centroids=centroids, minimum_distances=append_minimum_distances_block, i=i+block_size, max_block_size=max_block_size)

def distances_to_nearest_centroid_j(points_block: np.array, centroids: np.array, minimum_distances_block: np.array, j: int, max_block_size: int) -> np.array:

    if j >= centroids.shape[0]:
        return minimum_distances_block
    
    block_size = min(centroids.shape[0] - j, max_block_size)

    centroids_block = centroids[j:j+block_size]

    distances_to_centroids = points_to_points_distances(points_a=points_block, points_b=centroids_block)

    current_smallest_distances_to_centroid = minimums(distances_to_centroids)

    new_minimum_distances_block = np.minimum(minimum_distances_block, current_smallest_distances_to_centroid)

    return distances_to_nearest_centroid_j(points_block=points_block, centroids=centroids, minimum_distances_block=new_minimum_distances_block, j=j+block_size, max_block_size=max_block_size)

def means_of_distances(points_a: np.array, points_b: np.array, max_block_size: int) -> np.array:

    return means_of_distances_i(points_a=points_a, points_b=points_b, means=np.array([]), i=0, max_block_size=max_block_size)

def means_of_distances_i(points_a: np.array, points_b: np.array, means: np.array, i: int, max_block_size: int) -> np.array:

    if i >= points_a.shape[0]:
        return means.reshape(1, points_a.shape[0])

    block_size = min(points_a.shape[0] - i, max_block_size)

    points_a_block = points_a[i:i+block_size]

    means_block = np.zeros(block_size)

    new_means_block = means_of_distances_j(points_a_block=points_a_block, points_b=points_b, means_block=means_block, j=0, max_block_size=max_block_size)

    append_means = np.append(means, new_means_block)

    return means_of_distances_i(points_a=points_a, points_b=points_b, means=append_means, i=i+block_size, max_block_size=max_block_size)

def means_of_distances_j(points_a_block: np.array, points_b: np.array, means_block: np.array, j: int, max_block_size: int) -> np.array:

    if j >= points_b.shape[0]:
        return means_block
    
    block_size = min(points_b.shape[0] - j, max_block_size)

    points_b_block = points_b[j:j+block_size]

    distances = points_to_points_distances(points_a=points_a_block, points_b=points_b_block)
    
    new_means_block = np.add(means_block, np.sum(distances, axis=1) / points_b.shape[0])

    return means_of_distances_j(points_a_block=points_a_block, points_b=points_b, means_block=new_means_block, j=j+block_size, max_block_size=max_block_size)


def minimum_distance_indices(points_a: np.array, points_b: np.array, max_block_size: int) -> np.array:

    return minimum_distance_indices_i(points_a=points_a, points_b=points_b, minimum_distances_indices=np.array([]), i=0, max_block_size=max_block_size)

def minimum_distance_indices_i(points_a: np.array, points_b: np.array, minimum_distances_indices: np.array, i: int, max_block_size: int) -> np.array:
    
    if i >= points_a.shape[0]:
        return minimum_distances_indices.reshape(1, points_a.shape[0])
    
    block_size = min(points_a.shape[0] - i,  max_block_size)
    
    points_a_block = points_a[i:i+block_size]

    minimum_distances_block = np.full((block_size), fill_value=np.inf)

    minimum_distances_indices_block = np.full((block_size), fill_value=-1)

    new_minimum_distances_indices_block = minimum_distance_indices_j(points_a_block=points_a_block, points_b=points_b, minimum_distances_indices_block=minimum_distances_indices_block, minimum_distances_block=minimum_distances_block, j=0, max_block_size=max_block_size)
    
    append_minimum_distances_indices = np.append(minimum_distances_indices, new_minimum_distances_indices_block)

    return minimum_distance_indices_i(points_a=points_a, points_b=points_b, minimum_distances_indices=append_minimum_distances_indices, i=i+block_size, max_block_size=max_block_size)


def minimum_distance_indices_j(points_a_block: np.array, points_b: np.array, minimum_distances_indices_block: np.array, minimum_distances_block: np.array, j: int, max_block_size: int) -> np.array:

    if j >= points_b.shape[0]:
        return minimum_distances_indices_block
    
    block_size = min(points_b.shape[0] - j, max_block_size)

    points_b_block = points_b[j:j+block_size]

    new_minimum_distances_indices_block, new_minimum_distances_block = minimum_distance_indices_worker(points_a_block=points_a_block, points_b_block=points_b_block, minimum_distances_indices_block=minimum_distances_indices_block, minimum_distances_block=minimum_distances_block, j=j)

    return minimum_distance_indices_j(points_a_block=points_a_block, points_b=points_b, minimum_distances_indices_block=new_minimum_distances_indices_block, minimum_distances_block=new_minimum_distances_block, j=j+block_size, max_block_size=max_block_size)

def minimum_distance_indices_worker(points_a_block: np.array, points_b_block: np.array, minimum_distances_indices_block: np.array, minimum_distances_block: np.array, j: int) -> tuple[np.array, np.array]:

        
    distances_for_block = points_to_points_distances(points_a=points_a_block, points_b=points_b_block)
    
    curr_minimum_distance_indices_block = np.argmin(distances_for_block, axis=1)

    curr_minimum_distances_block = distances_for_block[np.arange(distances_for_block.shape[0]), curr_minimum_distance_indices_block]

    new_minimum_distances_indices_block = np.where(curr_minimum_distances_block < minimum_distances_block, curr_minimum_distance_indices_block + j, minimum_distances_indices_block)

    new_minimum_distances_block = np.minimum(minimum_distances_block, curr_minimum_distances_block)

    return new_minimum_distances_indices_block, new_minimum_distances_block


def initialize_centroids(data: np.array, number_of_centroids: int) -> np.array:

    # Initialize the first centroid to the point that has the shortest distance to all other points.
    init_centroids = np.array([single_point_in_densest_area(points_a=data, points_b=data, max_block_size=4000)])

    # Initialize remaining number of centroids.
    return initialize_centroids_helper(data=data, centroids=init_centroids, k=1, number_of_centroids=number_of_centroids)


def initialize_centroids_helper(data: np.array, centroids: np.array, k: int, number_of_centroids: int) -> np.array:

    if k == number_of_centroids:
        return  centroids
    
    # (n-points,) matrix where each value is the shortest distance to a centroid.
    minimum_distances = distances_to_nearest_centroid(points=data, centroids=centroids, max_block_size=4000)

    # Sort distances in descending order.
    sorted_minimum_distances = np.argsort(minimum_distances)[::-1]

    # Select some percentile of the furthest points.
    top_percentile_indices = sorted_minimum_distances[:max(int((0.01/k) * len(sorted_minimum_distances)), 1)]

    # From the refined list of points, find the one nearest to all points.
    next_centroid = single_point_in_densest_area(points_a=data[top_percentile_indices], points_b=data, max_block_size=4000)

    new_centroids = np.append(centroids, next_centroid).reshape(1, centroids.shape[1]+1)

    return initialize_centroids_helper(data=data, centroids=new_centroids, k=k+1, number_of_centroids=number_of_centroids)

def move_centroids(data: np.array, labels: np.array, new_centroids: np.array, k: int, number_of_centroids: int) -> np.array:

    # Move centroids based on mean of corresponding cluster.

    if k == number_of_centroids:
        return new_centroids.reshape(1, number_of_centroids)
    
    append_centroid = np.append(new_centroids, data[labels == k].mean(axis=0))

    return move_centroids(data=data, labels=labels, new_centroids=append_centroid, k=k+1, number_of_centroids=number_of_centroids)


def train(data: np.array, number_of_centroids: int, iteration: int, max_iterations: int) -> tuple[np.array, np.array]:

    # Train data by moving centroids until convergence or max_iterations is reached:
    
    init_centroids = initialize_centroids(data=data, number_of_centroids=number_of_centroids)

    init_labels = minimum_distance_indices(points_a=data, points_b=init_centroids, max_block_size=4000)
    
    return train_helper(data=data, centroids=init_centroids, labels=init_labels, iteration=iteration+1, max_iterations=max_iterations)

def train_helper(data: np.array, centroids: np.array, labels: np.array, iteration: int, max_iterations: int) -> tuple[np.array, np.array]:

    # Train data by moving centroids until convergence or max_iterations is reached:

    if iteration == max_iterations:
        return centroids, labels
    
    new_centroids = move_centroids(data=data, labels=labels, new_centroids=np.array([]), k=0, number_of_centroids=centroids.shape[0])
    
    new_labels = minimum_distance_indices(points_a=data, points_b=new_centroids, max_block_size=4000)

    if np.all(centroids == new_centroids): 
        return new_centroids, new_labels

    return train_helper(data=data, centroids=new_centroids, labels=new_labels, iteration=iteration+1, max_iterations=max_iterations)


def inference(data: np.array, centroids: np.array) -> np.array:

    return minimum_distance_indices(points_a=data, points_b=centroids, max_block_size=4000)

'''
def silhouette_score(data: np.array, labels: np.array, centroids: np.array) -> float:
    silhouette_scores = np.array([])

    for k in range(centroids.shape[0]):

        current_data_points = data[labels==k]

        a_i = means_of_distances(points_a=current_data_points, points_b=current_data_points, max_block_size=4000)

        other_clusters = np.array([data[labels==l] for l in range(centroids.shape[0]) if k != l], dtype=object)

        avg_distance_to_other_clusters = np.array([means_of_distances(points_a=current_data_points, points_b=other_cluster, max_block_size=4000) for other_cluster in other_clusters])

        b_i = avg_distance_to_other_clusters[np.argmin(avg_distance_to_other_clusters, axis=0), np.arange(avg_distance_to_other_clusters.shape[1])]

        s_i = (b_i - a_i) / (np.maximum(a_i, b_i))
        print(f"a_i {a_i}")
        print(f"b_i {b_i}")
        print(f"maximum {np.maximum(a_i, b_i)}")
        print(f"s_i {s_i}")

        silhouette_scores = np.append(silhouette_scores, s_i)

    return np.mean(silhouette_scores)

def optimal_train(data: np.array, start_k: int = 2, max_k: int = 100) -> tuple[np.array, np.array, int, float]:
    
    centroids_array = []
    labels_array = []
    silhouette_scores = []

    for k in range(start_k, max_k):
    
        centroids, labels = train(data=data, number_of_centroids=k, iteration=0, max_iterations=100)
        
        centroids_array.append(centroids)

        labels_array.append(labels)

        silhouette_scores.append(silhouette_score(data=data, labels=labels, centroids=centroids))

    optimal_index = np.argmax(silhouette_scores)
    optimal_k = optimal_index + start_k
    optimal_centroids  = centroids_array[optimal_index]
    optimal_labels = labels_array[optimal_index]
    optimal_silhouette_score = silhouette_scores[optimal_index]
    plot_clusters(data=data, centroids=optimal_centroids, labels=optimal_labels, silhouette_score=optimal_silhouette_score)

    return optimal_centroids, optimal_labels, optimal_k, optimal_silhouette_score
'''

# creating data
mean_01 = np.array([0.0, 0.0])
cov_01 = np.array([[1, 0.3], [0.3, 1]])
dist_01 = np.random.multivariate_normal(mean_01, cov_01, 100)

mean_02 = np.array([6.0, 7.0])
cov_02 = np.array([[1.5, 0.3], [0.3, 1]])
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 100)

mean_03 = np.array([7.0, -5.0])
cov_03 = np.array([[1.2, 0.5], 
                   [0.5, 1]]) 
dist_03 = np.random.multivariate_normal(mean_03, cov_01, 100)

mean_04 = np.array([2.0, -7.0])
cov_04 = np.array([[1.2, 0.5], [0.5, 1.3]])
dist_04 = np.random.multivariate_normal(mean_04, cov_01, 100)

data = np.vstack((dist_01, dist_02, dist_03, dist_04))
np.random.shuffle(data)


def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color='red', label='next centroid')
    plt.title('Select % d th centroid' % (centroids.shape[0]))

    plt.legend()
    plt.xlim(-5, 12)
    plt.ylim(-10, 15)
    plt.show()

def plot_clusters(data: np.array, centroids: np.array, labels: np.array, silhouette_score: float):

    number_of_clusters = centroids.shape[0]

    colors = cm.rainbow(np.linspace(0, 1, number_of_clusters))

    for k in range(centroids.shape[0]):
        current_data_points = data[labels == k]
        plt.scatter(current_data_points[:, 0], current_data_points[:, 1], marker='.', color=colors[k], label=f'cluster{k}')

    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', label='centroids')

    plt.title(f"Clusters with a Silhouette Score of {silhouette_score}")

    plt.legend()
    plt.xlim(-5, 20)
    plt.ylim(-10, 15)
    plt.show()

centroids, labels = train(data=data, number_of_centroids=5, iteration=0, max_iterations=100)


