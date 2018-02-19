import numpy as np
import math

SEQUENCE_SIZE = 10 # Sequence of points needed to extract features 



def reduce_points(points):
    assert SEQUENCE_SIZE < points.shape[0]
    reduced_points = []
    division = points.shape[0] / SEQUENCE_SIZE 

    return np.array([ points[x * division] for x in range(SEQUENCE_SIZE)])


def get_bounding_rectangles(points):
    # x_min, y_min, x_max, y_max
    return np.min(points[:, 0:1]), np.min(points[:, 1:]), np.max(points[:, 0:1]), np.max(points[:, 1:])

def get_center_and_radius(x_min, y_min, x_max, y_max):
    diagonal_distance = distance((x_min, y_min), (x_max, y_max))
    center = ((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)
    return center, diagonal_distance / 2

def normalize_size(points, center, radius):
    assert points.shape[0] > 1 and points.shape[1] == 2 ## Point Array should be of size N X 2
    assert len(center) == 2 ## center has x and y coordinate values 

    def get_distance_from_center(x):
        return int(math.sqrt(math.pow(x[0] - center[0], 2) + math.pow(x[1] -center[1], 2)) / radius)
    
    normalized_distances = np.apply_along_axis(get_distance_from_center, 1, points)
    return normalized_distances.flatten()

def normalize_angle(points, center, radius):
    assert points.shape[0] > 1 and points.shape[1] == 2 ## Point Array should be of size N X 2
    assert len(center) == 2 ## center has x and y coordinate values 
    initial_point = points[0]
    vector_initial = (initial_point[0] - center[0], initial_point[1] - center[1])

    # Angle is calculated relative to the line from 
    # Using the cosine law 
    # https://math.stackexchange.com/questions/361412/finding-the-angle-between-three-points

    def get_angle_from_center(x):

        ## TODO: Refactor this code with numpy
        vector = (x[0] - center[0], x[1] - center[1])
        
        distance_product = distance(vector) * distance(vector_initial)
        cosine_value = (vector[0] * vector_initial[0] + vector[1] * vector_initial[1]) / distance_product
        if vector == vector_initial:
            return 0
        return math.degrees(math.acos(cosine_value)) ## Not sure if this should be sin or cosine 

    
    normalized_angles = np.apply_along_axis(get_angle_from_center, 1, points)
        
    return normalized_angles.flatten()

def distance(x, y=(0, 0)):
    return math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2))

def extract_features(points):
    points = reduce_points(points)
        
    x_min, y_min, x_max, y_max = get_bounding_rectangles(points)
    center, radius =  get_center_and_radius(x_min, y_min, x_max, y_max)
    
    angles, distances = normalize_angle(points, center, radius), normalize_size(points, center, radius)
    
    features = np.append(angles, distances)

    return features



if __name__ == '__main__':
    pass