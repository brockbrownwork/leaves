import alphashape
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("...loaded modules.")

# Define a set of 2D points
points_2d = [(0., 0.), (0., 1.), (1., 1.), (1., 0.),
          (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]
# points_2d = list([(x * 100, y * 100) for (x, y) in points_2d])
# print(points_2d)

def generate_alpha_shape(points_2d, alpha):
    # Generate the alphashape with an alpha value of 2.0
    alpha_shape = alphashape.alphashape(points_2d, alpha)
    if alpha_shape.is_empty:
        return []
    # Get the coordinates of the points that make up the alphashape
    alphashape_coords = alpha_shape.exterior.coords.xy

    # Plot the original set of points and the alphashape
    fig, ax = plt.subplots()
    ax.scatter(*zip(*points_2d))
    ax.plot(alphashape_coords[0], alphashape_coords[1])
    plt.show()

    # print(alphashape_coords)
    return [(x, y) for (x, y) in zip(alphashape_coords[0], alphashape_coords[1])]

def normalize_points(points, original_bounds, new_bounds=(0, 1)):
    min_x, min_y = original_bounds[0]
    max_x, max_y = original_bounds[1]
    new_min, new_max = new_bounds
    return [((x - min_x) / (max_x - min_x) * (new_max - new_min) + new_min,
             (y - min_y) / (max_y - min_y) * (new_max - new_min) + new_min) for x, y in points]

def denormalize_points(points, original_bounds, new_bounds=(0, 1)):
    min_x, min_y = original_bounds[0]
    max_x, max_y = original_bounds[1]
    new_min, new_max = new_bounds
    return [((x - new_min) * (max_x - min_x) / (new_max - new_min) + min_x,
             (y - new_min) * (max_y - min_y) / (new_max - new_min) + min_y) for x, y in points]




# test it out
if __name__ == "__main__":
    original_bounds = [(0, 0), (100, 100)]

    scaled_points = [(x * 100, y * 100) for (x, y) in points_2d]
    normalized_points = normalize_points(scaled_points, original_bounds)
    alpha_shape_points = generate_alpha_shape(normalized_points, 2)
    denormalized_alpha_shape_points = denormalize_points(alpha_shape_points, original_bounds)

    print(denormalized_alpha_shape_points)
