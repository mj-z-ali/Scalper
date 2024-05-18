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
    minimum_indices = np.ma.argmin(values_array, axis=1)

    # (n-points,) matrix of smallest values.
    return values_array[np.arange(values_array.shape[0]), minimum_indices]

def points_to_points_distances(points_a: np.array, points_b: np.array) -> np.array:

    # Reshape to enable broadcasting.
    # Results in a (n, 1, d) matrix, s.t.:
    # n <- number of points in points_a,
    # d <- number of dimensions in both points_a and points_b.
    # Middle 1 dimension allows broadcasting across all points.
    points_expanded =  points_a[:, np.newaxis, :]

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

    
def point_nearest_to_most(points_a: np.array, points_b: np.array) -> np.array:

    # Total distances of each point to all other points.
    # Results in a (n-points_a,) matrix.
    distance_sums = np.sum(points_to_points_distances(points_a=points_a, points_b=points_b), axis=1)

    # Find point with smallest total distance.
    point_with_minimum_sum = points_a[np.argmin(distance_sums)] 

    return point_with_minimum_sum

def initialize_centroids(points: np.array, k: int) -> np.array:

    # Initialize the first centroid to the point that has the shortest distance to all other points.
    centroids = np.array([point_nearest_to_most(points_a=points, points_b=points)])

    plot(points, centroids)

    for i in range(1, k):
        # (n-points, m-centroids) matrix where each value is the distance from a point to a centroid.
        distances = points_to_points_distances(points_a=points, points_b=centroids)

        # (n-points,) matrix where each value is the shortest distance to a centroid.
        minimum_distances = minimums(values_array=distances)

        # Sort distances in descending order.
        sorted_minimum_distances = np.argsort(minimum_distances)[::-1]

        # Select some percentile of the furthest points.
        top_percentile_indices = sorted_minimum_distances[:max(int((0.25/i) * len(sorted_minimum_distances)), k)]
        
        # From the refined list of points, find the one nearest to all points.
        next_centroid = [point_nearest_to_most(points_a=points[top_percentile_indices], points_b=points)]

        centroids = np.append(centroids, next_centroid, axis=0)

        plot(points, centroids)

    return centroids

def move_centroids(data: np.array, centroids: np.array, nearest_centroid_indices: np.array) -> np.array:

    new_centroids = np.array([data[nearest_centroid_indices == k].mean(axis=0) for k in range(centroids.shape[0])])

    return new_centroids

def k_means(data: np.array, k: int, max_iterations: int = 100) -> tuple[np.array, np.array]:

    centroids = initialize_centroids(points=data, k=k)

    for i in range(max_iterations):
        # Distances of points to centroids.
        # Results in a (n-points, m-centroids) matrix.
        distances = points_to_points_distances(points_a=data, points_b=centroids)

        nearest_centroid_indices = np.argmin(distances, axis=1)

        print(f"nearest centroid indices{nearest_centroid_indices}")
        print(f"nearest centroid indices shape {nearest_centroid_indices.shape}")

        new_centroids = move_centroids(data=data, centroids=centroids, nearest_centroid_indices=nearest_centroid_indices)

        print(f"new centroids {new_centroids}")
        print(f"new centroids shape {new_centroids.shape}")

        if np.all(centroids == new_centroids):
            print(f"Iterations: {i}")
            break

        centroids = new_centroids

    return centroids, nearest_centroid_indices


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

def plot_clusters(data: np.array, centroids: np.array, nearest_centroid_indices: np.array):

    number_of_clusters = centroids.shape[0]

    colors = cm.rainbow(np.linspace(0, 1, number_of_clusters))

    for k in range(centroids.shape[0]):
        current_data_points = data[nearest_centroid_indices == k]
        plt.scatter(current_data_points[:, 0], current_data_points[:, 1], marker='.', color=colors[k], label=f'cluster{k}')

    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', label='centroids')

    plt.title('Clusters')

    plt.legend()
    plt.xlim(-5, 12)
    plt.ylim(-10, 15)
    plt.show()

# call the initialize function to get the centroids
centroids, nearest_centroid_indices = k_means(data=data,k=4)

plot_clusters(data=data, centroids=centroids, nearest_centroid_indices=nearest_centroid_indices)

