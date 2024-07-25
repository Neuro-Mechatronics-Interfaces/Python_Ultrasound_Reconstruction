import os
import numpy as np
from trc import TRCData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stlview import Object

def compute_matrix_from_markers(origin, side, front):
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

def draw_probe(ax, r, t, probe_model=None, probe_offset=None):
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
        if probe_offset is not None:
            t_probe = probe_offset
        else:
            t_probe = np.array([0, 0, 0])

        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t

        # Create a translation matrix for the 65 units shift on the x-axis
        T_probe = np.eye(4)
        T_probe[:3, 3] = t_probe

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

def update_figure(ax, r, t, probe_model=None, probe_offset=None):
    """ Draw the coordinate axes and vectors on the plot.

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

    # Define the axes and apply rotation
    scale = 50
    axes = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
    axes = np.dot(r, axes.T).T

    # Draw the axes
    ax.clear()  # Clear the previous arrows
    for color, axis in zip(['r', 'g', 'b'], axes):
        ax.quiver(*t, *axis, color=color)

    # Draw probe
    draw_probe(ax, r, t, probe_model, probe_offset)

    # Set the plot limits
    ax.set_xlim([t[0] - 150, t[0] + 150])
    ax.set_ylim([t[1] - 150, t[1] + 150])
    ax.set_zlim([t[2] - 150, t[2] + 150])

    # Update text and plot
    ax.text2D(0.02, 0.95, f"x:{t[0]}\ny:{t[1]}\nz:{t[2]}", transform=ax.transAxes)
    euler = R.from_matrix(r).as_euler('xyz', degrees=True)
    ax.text2D(0.32, 0.95, f"Roll: {euler[0]:.2f}\nPitch: {euler[1]:.2f}\nYaw: {euler[2]:.2f}", transform=ax.transAxes)
    plt.draw()

def get_probe_pose(file_path, cor_frame, show_animation=False, probe_model=None, probe_offset=None):
    """ Calculate the pose of the probe from the marker positions in the TRC file."""

    # Load the TRC file
    trc_data = TRCData()
    trc_data.load(file_path)

    # Get the rate and dataset points from the markers
    origin = np.array(trc_data['Origin'])
    side = np.array(trc_data[cor_frame['y']])
    front = np.array(trc_data[cor_frame['x']])
    mocap_rate = trc_data['OrigDataRate']  # Also want the rate for optional rendering

    # Set up figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    T_T_W = []  # Transformation from target frame {T} to world frame {W}
    for origin_marker, side_marker, front_marker in zip(origin, side, front):
        if np.isnan(origin_marker).any() or np.isnan(side_marker).any() or np.isnan(front_marker).any():
            continue
        
        # Compute the transformation matrix from the markers
        R_T_W, t_T_W = compute_matrix_from_markers(origin_marker, side_marker, front_marker)

        # Save pose
        T_T_W.append(np.r_[np.c_[R_T_W, t_T_W], [[0, 0, 0, 1]]])

        # Visualize results
        if show_animation:
            update_figure(ax, R_T_W, t_T_W, probe_model, probe_offset)
            plt.pause(1 / mocap_rate)  # Pause to update the plot

    print(f"Probe pose calculated for {len(T_T_W)} frames.")

    return T_T_W


# create main function
if __name__ == '__main__':

    print("This example takes the XYZ coordinates of 3 markers recorded by a motion capture system and computes the pose of an object in the same coordinate system.")

    # We define the path to the folder with the mocap dataset (either .trc or .c3d). The root directory needs to be
    # where the dataset folder is located.
    mocap_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\mocap\Session1\Recording001.trc'
    root_dir = r'C:\Users\HP\Documents\Github\Ultrasound\Python_Ultrasound_Reconstruction\dataset'

    # The marker pose data contained in the .trc file are 'Origin', 'Side', and 'Front'. We can use these to calculate
    # the homogenous transformation matrix of the object with respect to the world (or motion capture room). Origin is
    # the marker which the vectors are defined from and on the xy plane.
    cor_frame = {'x': 'Front', 'y': 'Side'}

    # We're also going to load the probe model to visualize it with the prober transformation
    # The offset from the origin point to the bottom center of the probe is (-65, -70, -115) mm
    probe_model_path = os.path.join(root_dir, '..', 'models', 'probe.stl')
    probe_model = Object(probe_model_path)
    probe_model.build()
    probe_offset = np.array([-65, -115, -70])  # z and y seem to be switched

    # Grab the transformation data, making sure to pass in the probe info as well.
    # Note: animating the probe model will slow down the process but confirms whether your markers are being referenced
    # correctly. I'd suggest setting it to True first to confirm the pose, then set it to False to speed up the process.
    show_animation = False  # Set to True to see the probe!
    T_T_W = get_probe_pose(mocap_path, cor_frame, show_animation=show_animation, probe_model=probe_model, probe_offset=probe_offset)

    # Save the combined transformation matrices data of the probe position to a .npy file
    np.save(os.path.join(root_dir, 'target_pose.npy'), np.stack(T_T_W, 0))
