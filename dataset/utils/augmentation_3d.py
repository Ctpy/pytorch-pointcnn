import numpy as np


def normalize_data(pc):
    """ Normalize the data, use coordinates of the block centered at origin,
        Input:
            NxC array
        Output:
            NxC array
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m

    return pc


def shuffle_points(pcl, labels):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Returns the shuffled pointcloud, labels and indices
        Input:
            NxC array, Nx1 array
        Output:
            NxC array, N array, N array
    """
    idx = np.arange(pcl.shape[0])
    np.random.shuffle(idx)
    return pcl[idx, :], labels[idx], idx


def rotate_point_cloud_z(pcl):
    """ Randomly rotate the point cloud to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    return np.dot(pcl, rotation_matrix)


def rotate_perturbation_point_cloud(pcl, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point cloud by small rotations
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    return np.dot(pcl, R)


def jitter_point_cloud(pcl, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, jittered point cloud
    """
    jittered_data = np.clip(np.random.normal(0, clip / 3, size=pcl.shape), -1 * clip, clip)
    return pcl + jittered_data


def shift_point_cloud(pcl, shift_range=0.1):
    """ Randomly shift point cloud.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, shifted point cloud
    """
    shift = np.random.uniform(-shift_range, shift_range, 3)
    return pcl + shift


def random_scale_point_cloud(pcl, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original point cloud
        Return:
            BxNx3 array, scaled point cloud
    """
    scale = np.random.uniform(scale_low, scale_high)
    return pcl * scale
