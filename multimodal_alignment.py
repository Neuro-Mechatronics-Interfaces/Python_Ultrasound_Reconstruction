# This script takes the ultrasound image data from the raw data folder and aligns it with the motion capture data.

import os
import cv2
import numpy as np
from trc import TRCData
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


# Load the ultrasound image data
def load_data(data_folder):
    data = []
    for file in os.listdir(data_folder):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(data_folder, file), cv2.IMREAD_GRAYSCALE)
            data.append(img)
    return data


# Path to the folder with the mocap data
mocap_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\mocap\Session1'
file_name = 'Recording002.trc'

# Load the trc file
trc_path = os.path.join(mocap_path, file_name)
trc_data = TRCData()
trc_data.load(trc_path)

# Get the rate and data points from the markers
mocap_rate = trc_data['OrigDataRate']
front_pos = np.array(trc_data['Front'])
mid_pos = np.array(trc_data['Mid'])
side_pos = np.array(trc_data['Side'])
origin_pos = np.array(trc_data['Origin'])

# Get the number of data points
N_points = front_pos.shape[0]

# get the timestamp of the data
t = np.arange(0, N_points) / mocap_rate
print(t.shape)

# =============================================================================
if False:
    # Initialize the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1000, 1000])  # Adjust x-axis limits if needed
    ax.set_ylim([-1000, 1000])  # Adjust y-axis limits if needed
    ax.set_zlim([-1000, 1000])  # Adjust z-axis limits if needed

    # Initialize the plot object, make the size smaller if needed
    points, = ax.plot([], [], [], marker='o', linestyle='None', markersize=1)

    # Function to update the plot for each frame of the animation
    def update_plot(frame):
        # print("displaying frame", frame)
        # print("Coordinates:", front_pos[frame, :])
        points.set_data(front_pos[:frame, 0], front_pos[:frame, 1])
        points.set_3d_properties(front_pos[:frame, 2])
        return points,

    # Create the animation. The rate of the animation should be at 150hz
    ani = animation.FuncAnimation(fig, update_plot, frames=N_points, interval=1000 / mocap_rate, blit=True)
    plt.show()

# =============================================================================

# Calculate velocity and acceleration
zero_pad = np.zeros((1, 3))
temp = np.diff(trc_data['Front'], axis=0)
vel = np.concatenate((zero_pad, temp), axis=0)

temp = np.diff(vel, axis=0)
acc = np.concatenate((zero_pad, temp), axis=0)

# Calculate the magnitude of the acceleration
acc_mag = np.linalg.norm(acc, axis=1)
plt.plot(t, acc_mag)
plt.title('Acceleration Magnitude')
plt.xlabel('Time (s)')
plt.show()

# Path to the folder with the ultrasound data
ultrasound_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\video'
video_filename = 'raw002.mp4'

cap = cv2.VideoCapture(os.path.join(ultrasound_path, video_filename))
if not cap.isOpened():
    print("Error opening video stream or file")

# If the video loads, great, now we can do some processing
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print("Video frame rate:", frame_rate)
N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of frames:", N_frames)
video_length = N_frames / frame_rate
print("Video length (s):", video_length)
prev_frame = None
us_pxdf = []
print("Processing video...")
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # We need to convert the 2D frames into a 1D signal where the pixel displacement in the images can correlate with
    # the acceleration magnitude of the motion capture data. This way we can synchronize the 2 data modalities.
    if prev_frame is not None:
        diff = cv2.absdiff(frame, prev_frame) # Gets the difference between each frame
        diff_sum = cv2.sumElems(diff) # Converts the 2D signal into a 1-by-4,
        us_pxdf.append(np.sum(diff_sum))# sum of all pixel values

    prev_frame = frame

    # Display the resulting frame
    #cv2.imshow('Frame', frame)
    # Press Q on keyboard to  exit
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    break

# When everything done, release the video capture object
print("Done")
cap.release()
cv2.destroyAllWindows()

us_pxdf = np.array(us_pxdf)
print("New shape of the ultrasound data:")
print(us_pxdf.shape)
# Show the 1D data
t_us = np.arange(0, us_pxdf.shape[0]) / frame_rate
print("New video length (s):", t_us[-1])
plt.plot(t_us, us_pxdf)
plt.show()

# Downsample the acc data to match the ultrasound data
def downsample_by_averaging(source_array, target_length):
    # Calculate the segment size
    segment_size = len(source_array) / target_length

    # Create the downsampled array
    downsampled_array = np.array([
        np.mean(source_array[int(i * segment_size):int((i + 1) * segment_size)])
        for i in range(target_length)
    ])

    return downsampled_array

acc_mag_ds = downsample_by_averaging(acc_mag, us_pxdf.shape[0])
t_ds = np.arange(0, acc_mag_ds.shape[0]) / frame_rate

# Perform cross-correlation to find the time shift between the 2 signals
cross_corr = np.correlate(us_pxdf, acc_mag_ds, mode='full')

# Compute the lag array
lags = np.arange(-len(acc_mag_ds) + 1, len(us_pxdf)) # Compute lag array
#lags = np.arange(-len(us_pxdf) + 1, len(acc_mag_ds)) # Compute lag array
max_corr_index = np.argmax(cross_corr) # Find the index of the maximum cross-correlation value
t_displacement = lags[max_corr_index] # Find the corresponding lag/time displacement
print("Time displacement (lag) between signals:", t_displacement)

# Plot the cross-correlation
plt.figure()
plt.title('Cross-Correlation')
plt.plot(lags, cross_corr)
plt.title('Cross-Correlation')
plt.xlabel('Lag')
plt.ylabel('Correlation')
#plt.axvline(x=t_displacement, color='r', linestyle='--')
plt.show()



