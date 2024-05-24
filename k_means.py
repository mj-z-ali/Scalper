import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def minimums_ignoring_zero(values_array: np.array) -> np.array:
    
    # Ignore zeros
    masked_array = np.ma.masked_equal(values_array, 0)

    # Retrieve indices of smallest values across the second dimension.
    minimum_indices = np.ma.argmin(masked_array, axis=1)

    # (n-points,) matrix of smallest values.
    return values_array[np.arange(values_array.shape[0]), minimum_indices]

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


def single_point_in_densest_area(points_a: np.array, points_b: np.array, block_size: int = 4000) -> np.array:

    distance_sums = np.zeros(points_a.shape[0])

    for i in range(0, points_a.shape[0], block_size):

        points_a_block = points_a[i:i+block_size]

        for j in range(0, points_b.shape[0], block_size):

            points_b_block = points_b[j:j+block_size]

            distance_sums_for_block = np.sum(points_to_points_distances(points_a=points_a_block, points_b=points_b_block), axis=1)

            distance_sums[i:i+block_size] += distance_sums_for_block
    
    return points_a[np.argmin(distance_sums)]

def distances_to_nearest_centroid(points: np.array, centroids: np.array, block_size: int = 4000) -> np.array:

    minimum_distances = np.full((points.shape[0]), fill_value=np.inf)

    for i in range(0, points.shape[0], block_size):

        points_block = points[i:i+block_size]

        for j in range(0, centroids.shape[0], block_size):

            centroids_block = centroids[j:j+block_size]

            distances_to_centroids = points_to_points_distances(points_a=points_block, points_b=centroids_block)

            current_smallest_distances_to_centroid = minimums(distances_to_centroids)

            minimum_distances[i:i+block_size] = np.minimum(minimum_distances[i:i+block_size], current_smallest_distances_to_centroid)

    return minimum_distances
    
def point_nearest_to_most(points_a: np.array, points_b: np.array) -> np.array:

    # Total distances of each point to all other points.
    # Results in a (n-points_a,) matrix.
    distance_sums = np.sum(points_to_points_distances(points_a=points_a, points_b=points_b), axis=1)

    # Find point with smallest total distance.
    point_with_minimum_sum = points_a[np.argmin(distance_sums)] 

    return point_with_minimum_sum

def initialize_centroids(points: np.array, k: int) -> np.array:

    # Initialize the first centroid to the point that has the shortest distance to all other points.
    centroids = np.array([single_point_in_densest_area(points_a=points, points_b=points)])

    for i in range(1, k):
        # (n-points, m-centroids) matrix where each value is the distance from a point to a centroid.
        distances = points_to_points_distances(points_a=points, points_b=centroids)

        # (n-points,) matrix where each value is the shortest distance to a centroid.
        minimum_distances = minimums(values_array=distances)

        minimum_distances_test = distances_to_nearest_centroid(points=points, centroids=centroids)

        print(f"test: {minimum_distances==minimum_distances_test}")
        # Sort distances in descending order.
        sorted_minimum_distances = np.argsort(minimum_distances)[::-1]

        # Select some percentile of the furthest points.
        top_percentile_indices = sorted_minimum_distances[:max(int((0.25/i) * len(sorted_minimum_distances)), k)]
        
        # From the refined list of points, find the one nearest to all points.
        next_centroid = [single_point_in_densest_area(points_a=points[top_percentile_indices], points_b=points)]

        centroids = np.append(centroids, next_centroid, axis=0)

    return centroids

def move_centroids(data: np.array, centroids: np.array, nearest_centroid_indices: np.array) -> np.array:

    new_centroids = np.array([data[nearest_centroid_indices == k].mean(axis=0) for k in range(centroids.shape[0])])

    return new_centroids

def k_means(data: np.array, k: int = 2, max_iterations: int = 100) -> tuple[np.array, np.array]:

    centroids = initialize_centroids(points=data, k=k)

    for i in range(max_iterations):
        # Distances of points to centroids.
        # Results in a (n-points, m-centroids) matrix.
        distances = points_to_points_distances(points_a=data, points_b=centroids)

        nearest_centroid_indices = np.argmin(distances, axis=1)

        new_centroids = move_centroids(data=data, centroids=centroids, nearest_centroid_indices=nearest_centroid_indices)


        if np.all(centroids == new_centroids):
            print(f"Iterations: {i}")
            break

        centroids = new_centroids

    return centroids, nearest_centroid_indices

def k_means_model(data: np.array, centroids: np.array) -> np.array:

    distances = points_to_points_distances(points_a=data, points_b=centroids)

    return np.argmin(distances, axis=1)

def silhouette_score(data: np.array, labels: np.array, centroids: np.array) -> float:
    silhouette_scores = np.array([])

    for k in range(centroids.shape[0]):

        current_data_points = data[labels==k]

        a_i = points_to_points_distances(points_a=current_data_points, points_b=current_data_points).mean(axis=1)

        other_clusters = np.array([data[labels==l] for l in range(centroids.shape[0]) if k != l], dtype=object)

        avg_distance_to_other_clusters = np.array([points_to_points_distances(points_a=current_data_points, points_b=other_cluster).mean(axis=1) for other_cluster in other_clusters])

        b_i = avg_distance_to_other_clusters[np.argmin(avg_distance_to_other_clusters, axis=0), np.arange(avg_distance_to_other_clusters.shape[1])]

        s_i = (b_i - a_i) / (np.maximum(a_i, b_i))

        silhouette_scores = np.append(silhouette_scores, s_i)

    return np.mean(silhouette_scores)

def k_means_fit(data: np.array, max_k: int = 100) -> tuple[np.array, np.array, int, float]:
    
    centroids_array = []
    labels_array = []
    silhouette_scores = []
    start_k = 2

    for k in range(start_k, max_k):
    
        centroids, labels = k_means(data=data,k=k)
        
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


# creating data
# mean_01 = np.array([0.0, 0.0])
# cov_01 = np.array([[1, 0.3], [0.3, 1]])
# dist_01 = np.random.multivariate_normal(mean_01, cov_01, 100)

# mean_02 = np.array([6.0, 7.0])
# cov_02 = np.array([[1.5, 0.3], [0.3, 1]])
# dist_02 = np.random.multivariate_normal(mean_02, cov_02, 100)

# mean_03 = np.array([7.0, -5.0])
# cov_03 = np.array([[1.2, 0.5], 
#                    [0.5, 1]]) 
# dist_03 = np.random.multivariate_normal(mean_03, cov_01, 100)

# mean_04 = np.array([2.0, -7.0])
# cov_04 = np.array([[1.2, 0.5], [0.5, 1.3]])
# dist_04 = np.random.multivariate_normal(mean_04, cov_01, 100)

# data = np.vstack((dist_01, dist_02, dist_03, dist_04))
# np.random.shuffle(data)

# Parameters for different clusters
mean_01 = 0.0
std_01 = 1.0  # Standard deviation
dist_01 = np.random.normal(mean_01, std_01, 100)

mean_02 = 6.0
std_02 = 1.5
dist_02 = np.random.normal(mean_02, std_02, 100)

mean_03 = 7.0
std_03 = 1.2
dist_03 = np.random.normal(mean_03, std_03, 100)

mean_04 = 2.0
std_04 = 1.2
dist_04 = np.random.normal(mean_04, std_04, 100)

mean_05 = 15.0
std_05 = 1
dist_05 = np.random.normal(mean_05, std_05, 50)

# Combine all distributions into a single dataset
data = np.concatenate((dist_01, dist_02, dist_03, dist_04, dist_05))
np.random.shuffle(data) 

zeros = np.zeros(data.shape[0])

# Stack them together
data_2d = np.column_stack((data, zeros))

print(data_2d)



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

centroids, labels, k, score = k_means_fit(data=data_2d, max_k=50)


