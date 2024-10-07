import numpy as np
import numba


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    return np.linalg.norm(t1 - t2)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def extract_regions(image, x_pixel, y_pixel, beta):
    beta_half = round(beta / 2)
    max_x, max_y = image.shape[:2]

    # Create meshgrid for x and y coordinates
    x = np.arange(2 * beta_half + 1) - beta_half
    y = np.arange(2 * beta_half + 1) - beta_half
    xx, yy = np.meshgrid(x, y)

    # Broadcast to match the number of pixels
    xx = xx[np.newaxis, :, :] + x_pixel[:, np.newaxis, np.newaxis]
    yy = yy[np.newaxis, :, :] + y_pixel[:, np.newaxis, np.newaxis]

    # Clip coordinates to image boundaries
    xx = np.clip(xx, 0, max_x - 1)
    yy = np.clip(yy, 0, max_y - 1)

    # Extract regions
    regions = image[xx, yy]

    return regions


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    g = extract_regions(image, x_pixel, y_pixel, beta).mean(axis=(1, 2))
    c0, c1, c2 = g[:, 0], g[:, 1], g[:, 2]
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / \
        (np.var(c0) + np.var(c1) + np.var(c2))
    return c3


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None,
                         beta=49, alpha=1, histology=True, z=None):
    # x,y,x_pixel, y_pixel are lists
    if histology or z is not None:
        assert (x_pixel is not None) & (y_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))
        print("Calculating adj matrix using histology image...")

        # If z is passed dynamically, use that instead
        c3 = z if z is not None else extract_color(x_pixel, y_pixel, image, beta)

        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculating adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)
