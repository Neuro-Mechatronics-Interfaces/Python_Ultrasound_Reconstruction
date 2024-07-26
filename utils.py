import re
import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============== General utilities ==============

def natsort(l):
    """
    Lambda function for nautural sorting of strings. Useful for sorting the
    list of file name of images with the target. Taken from:
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/

    input:
        l: list of input images with the target
    output:
        Nutural sorted list of images
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)

def downsampleByAveraging(source_array, target_length):
    # Calculate the segment size
    segment_size = len(source_array) / target_length

    # Create the downsampled array
    downsampled_array = np.array([
        np.mean(source_array[int(i * segment_size):int((i + 1) * segment_size)])
        for i in range(target_length)
    ])
    return downsampled_array

def computeAccelerationMagnitude(samples):
    """ Helper function for calculating the acceleration magnitude between matrices
    """
    translations = [sample[:3, 3] for sample in samples]  # Extract translation vectors
    velocities = np.diff(translations, axis=0)  # Compute velocities
    accelerations = np.diff(velocities, axis=0)  # Compute accelerations
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)  # Compute magnitudes
    return acceleration_magnitudes


# ======== Pose and transformation utilities =======

def computeMatrixFromMarkers(origin, side, front):
    """
    Calculate the transformation matrix of the coordinate frame defined by the marker positions.

    Parameters:
    ----------------
        origin (array): XYZ coordinates of the origin marker
        side (array): XYZ coordinates of the side marker
        front (array): XYZ coordinates of the front marker

    Returns:
    ----------------
        rotation_matrix (array): Rotation matrix of the coordinate frame
        origin (array): XYZ coordinates of the origin marker
    """

    # if any of the lists contain nan values, return list of zeros
    if np.isnan(origin).any() or np.isnan(side).any() or np.isnan(front).any():
        return np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))

    # Define the origin
    origin = np.array(origin)

    # Define the X-axis (vector from Origin to Front)
    x_axis = np.array(front) - origin
    x_axis = x_axis / np.linalg.norm(x_axis)  # Normalize to unit vector

    # Define the Y-axis (vector from Origin to Side)
    y_axis = np.array(side) - origin
    y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize to unit vector

    # Define the Z-axis (cross product of X and Y axes)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)  # Normalize to unit vector

    # Ensure orthogonality by recalculating Y-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize to unit vector

    # Rotation matrix (transformation matrix without translation)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis])

    return rotation_matrix, origin

def downsampleMatrixData(pose_data, target_length):
    """
    Downsample the pose data to match the target length.

    Parameters:
    ----------------
    pose_data : list of np.ndarray
        List of 4x4 transformation matrices representing the pose data.
    target_length : int
        The target length to downsample to (number of ultrasound frames).

    Returns:
    ----------------
    downsampled_pose_data : list of np.ndarray
        List of downsampled 4x4 transformation matrices.
    """
    current_length = len(pose_data)
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    downsampled_pose_data = np.array([pose_data[i] for i in indices])
    return downsampled_pose_data

# ========== Image processing functions ==========

def loadImages(us_img_list, flip=False):
    """ Load ultrasound images from a list of file paths.

    Parameters
    ----------
    us_img_list : list
        List of file paths to the ultrasound images.
    flip : bool
        Flag to flip the images vertically.

    Returns
    -------
    us_images : list
        List of ultrasound images.
    """

    us_images = []
    for img in us_img_list:
        frame = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if frame is not None:

            if flip:
                # Flip the image vertically
                frame = np.flipud(frame)

            us_images.append(frame)

    return us_images

def getPixelDisplacement(imlist_us, show_video):
    """ Computes the pixel displacement between frames

    Parameters:
    ----------
    imlist_us : (List) of ultrasound images
    show_video : (bool) Flag to show the ultrasound video

    Returns:
    --------
    us_px_diff : (List) of pixel displacements between frames
    """
    print("Calculating pixel diplacement...", end="")
    us_px_diff = []
    prev_frame = None
    for img in imlist_us:
        frame = cv2.imread(img, 0)

        # We need to convert the 2D frames into a 1D signal where the pixel displacement in the images can correlate
        # with the acceleration magnitude of the pose dataset. This way we can synchronize the 2 modalities.
        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)  # Gets the difference between each frame
            diff_sum = cv2.sumElems(diff)  # Converts the 2D signal into a 1-by-4,
            us_px_diff.append(np.sum(diff_sum))  # sum of all pixel values

        prev_frame = frame

        if show_video:
            cv2.imshow('Ultrasound', frame)
            # Press Esc or q to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if cv2.waitKey(255) == 27:
                break

    us_px_diff = np.array(us_px_diff) / np.max(us_px_diff)  # Normalize the pixel displacement signal
    print("done")
    return us_px_diff

def cropFrame(frame, crop_rect):
    """Crop the frame to the specified rectangle."""
    x, y, w, h = crop_rect
    return frame[y:y+h, x:x+w]

def plotImage(ax, img, T):
    """Plot an image on the 3D axes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
    img : np.ndarray
        Image to be plotted.
    T : np.ndarray
        Homogeneous transformation matrix.

    Returns
    -------
        None
    """

    # Interpolate the image coordinates in 3D
    h, w = img.shape
    xx = np.zeros((h, w))
    yy = np.zeros((h, w))
    zz = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            point = np.array([j, i, 0, 1])
            transformed_point = T @ point
            xx[i, j] = transformed_point[0]
            yy[i, j] = transformed_point[1]
            zz[i, j] = transformed_point[2]

    # Normalize the grayscale image to the range [0, 1] and then stack into RGB
    img_normalized = img / 255.0
    img_rgb = np.stack([img_normalized] * 3, axis=-1)

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img_rgb, shade=False)
    return None

def otsuThresholding(image):
    """Perform Otsu's thresholding to segment an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image to be segmented.

    Returns
    -------
    np.ndarray
        Binary segmented image.

    """
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def kmeansSegmentation(image, K=2):
    """Perform K-means clustering to segment an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or color image to be segmented.

    K : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Segmented image.

    """
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    segmented_image = res.reshape((image.shape))

    return segmented_image

def sharpenImage(image):
    """Sharpen an image using a kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Sharpened image.

    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def convertToGrayscale(image):
    """Convert an image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Grayscale image.

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#  ======== Plotting and drawing Utilities ========

def drawModel(ax, r, t, probe_model=None, probe_offset=None):
    """ Draw the probe model on the plot.

    Parameters:
    ----------------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object to draw the axes on.
    R : np.ndarray
        The rotation matrix.
    t : np.ndarray
        The translation vector.
    probe_model : Object
        The object to draw in the plot.
    """

    if probe_model is not None:

        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t

        # Create a translation matrix for the 65 units shift on the x-axis
        T_probe = np.eye(4)
        T_probe[:3, 3] = probe_offset

        # Combine the transformations
        T_combined = T @ T_probe

        # Plot the probe model if provided
        faces = []
        for face in probe_model.data:
            v0 = np.dot(T_combined, np.append([face._v1.x, face._v1.y, face._v1.z], 1))[:3]
            v1 = np.dot(T_combined, np.append([face._v2.x, face._v2.y, face._v2.z], 1))[:3]
            v2 = np.dot(T_combined, np.append([face._v3.x, face._v3.y, face._v3.z], 1))[:3]

            faces.append([v0, v1, v2])
        probe_poly = Poly3DCollection(faces, color='gray', alpha=0.5)
        ax.add_collection3d(probe_poly)

        # Also want to add a marker to the bottom center of the probe model, so we know where the ultrasound images will
        # be centered from (assuming the probe piece size is 50mm x 35mm)
        T_us_center = np.eye(4)
        T_us_center[:3, 3] = np.array([50/2, -1, 35/2])
        T_marker = T_combined @ T_us_center

        # Plot the marker
        ax.scatter(*T_marker[:3, 3], color='r', marker='o', s=100)

# ========== 3D reconstruction utilities ==========

def writePointCloud(X, C, file_name):
    n = X.shape[1]
    X = np.concatenate([X.T.astype(np.float32)] + 3 * [C[:, None]], 1)

    template = "%.4f %.4f %.4f %d %d %d\n"
    template = n * template
    data = template % tuple(X.ravel())

    with open(file_name, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        f.write(data)
