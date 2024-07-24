import os
import cv2
import numpy as np
import re
import pandas as pd
from trc import TRCData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def get_number_from_filename(file_path):
    # Create a new folder with the file number tag
    number_match = re.search(r'(\d+)(?=\.\w+$)', file_path)
    if number_match:
        number = number_match.group()
    else:
        raise ValueError("No number found in the file name")
    return number


def get_processed_data_folder(number):
    # Check the current and previous directory for the data folder and make "processed_data" folder
    processed_data_folder = ''
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        if not os.path.exists(os.path.join(os.getcwd(), 'data', 'processed_{}'.format(number))):
            os.makedirs(os.path.join(os.getcwd(), 'data', 'processed_{}'.format(number)))
        processed_data_folder = os.path.join(os.getcwd(), 'data', 'processed_{}'.format(number))
    elif os.path.exists(os.path.join(os.getcwd(), '..', 'data')):
        if not os.path.exists(os.path.join(os.getcwd(), '..', 'data', 'processed_{}'.format(number))):
            os.makedirs(os.path.join(os.getcwd(), '..', 'data', 'processed_{}'.format(number)))
        processed_data_folder = os.path.join(os.getcwd(), '..', 'data', 'processed_{}'.format(number))
    else:
        raise ValueError("No data folder found")
    return processed_data_folder


def get_probe_pose_example(trc_path, show_trc_animation=False):

    # Load the ultrasound image data
    def load_data(data_folder):
        data = []
        for file in os.listdir(data_folder):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(data_folder, file), cv2.IMREAD_GRAYSCALE)
                data.append(img)
        return data

    trc_data = TRCData()
    trc_data.load(trc_path)

    # Get the rate and data points from the markers
    mocap_rate = trc_data['OrigDataRate']
    print("Mocap rate: ", mocap_rate)
    origin = np.array(trc_data['Origin'])
    side = np.array(trc_data['Side'])
    front = np.array(trc_data['Front'])

    # Get the number of data points
    N_points = origin.shape[0]

    # get the timestamp of the data
    t_mocap = np.arange(0, N_points) / mocap_rate
    print(t_mocap.shape)
    print("Mocap recording length (s): ", t_mocap[-1])

    if show_trc_animation:
        # Initialize the figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()  # Turn on interactive mode

    euler_angles = np.zeros((N_points, 3))
    x_axis = np.zeros((N_points, 3))
    y_axis = np.zeros((N_points, 3))
    z_axis = np.zeros((N_points, 3))
    for i in range(N_points):
        # Get the euler angles for each frame
        euler_angles[i], x_axis[i], y_axis[i], z_axis[i] = calculate_euler_angles(origin[i], side[i], front[i])

        if x_axis[i].all() == 0 or y_axis[i].all() == 0 or z_axis[i].all() == 0:
            continue

        if show_trc_animation:
            ax.clear()  # Clear the previous arrows
            ax.quiver(origin[i][0], origin[i][1], origin[i][2], 50*x_axis[i][0], 50*x_axis[i][1], 50*x_axis[i][2], color='r', label='X-axis')
            ax.quiver(origin[i][0], origin[i][1], origin[i][2], 50*y_axis[i][0], 50*y_axis[i][1], 50*y_axis[i][2], color='g', label='Y-axis')
            ax.quiver(origin[i][0], origin[i][1], origin[i][2], 50*z_axis[i][0], 50*z_axis[i][1], 50*z_axis[i][2], color='b', label='Z-axis')

            # Setting the plot limits to be at origin
            ax.set_xlim([origin[i][0] - 100, origin[i][0] + 100])
            ax.set_ylim([origin[i][1] - 100, origin[i][1] + 100])
            ax.set_zlim([origin[i][2] - 100, origin[i][2] + 100])

            # Add text
            ax.text2D(0.02, 0.95, f"Euler Angles:\nRoll: {euler_angles[i][0]:.2f}\nPitch: {euler_angles[i][1]:.2f}\nYaw: {euler_angles[i][2]:.2f}",
                      transform=ax.transAxes)
            ax.text2D(0.32, 0.95, f"Origin position:\nX: {origin[i][0]:.2f}\nY: {origin[i][1]:.2f}\nZ: {origin[i][2]:.2f}",
                      transform=ax.transAxes)
            ax.text2D(0.62, 0.95, f"Mocap rate: {mocap_rate} Hz", transform=ax.transAxes)

            plt.draw()
            plt.pause(1/mocap_rate)  # Pause to update the plot

    # Save the pose data to the processed folder
    number = get_number_from_filename(trc_path)
    processed_data_folder = get_processed_data_folder(number)

    data = np.hstack((t_mocap.reshape(-1, 1), origin, euler_angles))
    pose_data = pd.DataFrame(data, columns=['Time', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'])
    pose_data.to_csv(os.path.join(processed_data_folder, 'pose_data.csv'), index=False)


def calculate_euler_angles(origin, side, front):
    """
    Calculate the Euler angles of the coordinate frame defined by the marker positions.

    Parameters:
    origin (array): XYZ coordinates of the origin marker
    side (array): XYZ coordinates of the side marker
    front (array): XYZ coordinates of the front marker

    Returns:
    euler_angles (array): Euler angles (in degrees) of the coordinate frame in the order 'xyz'
    """

    # Show the original data points
    #print("Origin:", origin)
    #print("Side:", side)
    #print("Front:", front)

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
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Convert the rotation matrix to Euler angles
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)

    return euler_angles, x_axis, y_axis, z_axis

def demo_pose():

    # Sample data: replace these with your actual coordinates
    #marker_data = {
    #    "origin": [-420.43924,  290.6915, -1094.7263],
    #    "side":   [-415.46475, 269.48877, -1098.3977],
    #    "front":  [-411.44675, 266.78574, -1102.9873]}
    origin = [-420.43924,  290.6915, -1094.7263]
    side = [-415.46475, 269.48877, -1098.3977]
    front = [-411.44675, 266.78574, -1102.9873]

    euler_angles, _, _, _ = calculate_euler_angles(origin, side, front)
    print("Euler Angles (degrees):", euler_angles)


# create main function
if __name__ == '__main__':

    print("This example takes the XYZ coordinates of 3 markers recorded by a motion capture system and computes the pose of an object in the same coordinate system.")

    # Path to the folder with the mocap data
    mocap_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\mocap\Session1'
    file_name = 'Recording002.trc'
    trc_path = os.path.join(mocap_path, file_name)

    get_probe_pose_example(trc_path, show_trc_animation=False)
    #demo_pose()
