# This script takes the ultrasound image dataset from the raw dataset folder and aligns it with the motion capture dataset.

import os
import re
import cv2
import numpy as np
from trc import TRCData
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def multimodal_alignment_example(trc_path, video_path,
                                 show_trc_animation=False,
                                 show_video=False,
                                 show_pixel_displacement=False,
                                 show_cross_corr=False,
                                 show_aligned_signals=False):

    # Load the ultrasound image dataset
    def load_data(data_folder):
        data = []
        for file in os.listdir(data_folder):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(data_folder, file), cv2.IMREAD_GRAYSCALE)
                data.append(img)
        return data

    trc_data = TRCData()
    trc_data.load(trc_path)

    # Create a new folder with the file number tag
    number_match = re.search(r'(\d+)(?=\.\w+$)', trc_path)
    if number_match:
        number = number_match.group()
    else:
        raise ValueError("No number found in the file name")

    # Check the current and previous directory for the dataset folder and make "processed_data" folder
    processed_data_folder = ''
    if os.path.exists(os.path.join(os.getcwd(), 'dataset')):
        if not os.path.exists(os.path.join(os.getcwd(), 'dataset', 'processed_{}'.format(number))):
            os.makedirs(os.path.join(os.getcwd(), 'dataset', 'processed_{}'.format(number)))
        processed_data_folder = os.path.join(os.getcwd(), 'dataset', 'processed_{}'.format(number))
    elif os.path.exists(os.path.join(os.getcwd(), '..', 'dataset')):
        if not os.path.exists(os.path.join(os.getcwd(), '..', 'dataset', 'processed_{}'.format(number))):
            os.makedirs(os.path.join(os.getcwd(), '..', 'dataset', 'processed_{}'.format(number)))
        processed_data_folder = os.path.join(os.getcwd(), '..', 'dataset', 'processed_{}'.format(number))
    else:
        raise ValueError("No dataset folder found")

    # Get the rate and dataset points from one marker
    mocap_rate = trc_data['OrigDataRate']
    front_pos = np.array(trc_data['Front'])

    # Get the number of dataset points
    N_points = front_pos.shape[0]


    # get the timestamp of the dataset
    t_mocap = np.arange(0, N_points) / mocap_rate
    print(t_mocap.shape)
    print("Mocap recording length (s): ", t_mocap[-1])

    if show_trc_animation:
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


    # Calculate velocity and acceleration
    zero_pad = np.zeros((1, 3))
    temp = np.diff(trc_data['Front'], axis=0)
    vel = np.concatenate((zero_pad, temp), axis=0)

    temp = np.diff(vel, axis=0)
    acc = np.concatenate((zero_pad, temp), axis=0)

    # Calculate the magnitude of the acceleration
    acc_mag = np.linalg.norm(acc, axis=1)

    # Load the ultrasound dataset
    cap = cv2.VideoCapture(video_path)
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
        # the acceleration magnitude of the motion capture dataset. This way we can synchronize the 2 dataset modalities.
        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame) # Gets the difference between each frame
            diff_sum = cv2.sumElems(diff) # Converts the 2D signal into a 1-by-4,
            us_pxdf.append(np.sum(diff_sum))# sum of all pixel values

        prev_frame = frame

        if show_video:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # When everything done, release the video capture object
    print("Done")
    cap.release()
    cv2.destroyAllWindows()

    us_pxdf = np.array(us_pxdf) / np.max(us_pxdf) # Normalize the pixel displacement signal
    print("New shape of the ultrasound dataset:")
    print(us_pxdf.shape)
    t_us = np.arange(0, us_pxdf.shape[0]) / frame_rate
    print("New video length (s):", t_us[-1])

    if show_pixel_displacement:
        plt.title('1D Pixel Displacement')
        plt.plot(t_us, us_pxdf)
        plt.xlabel('Time (s)')
        plt.ylabel('Pixel Displacement')
        plt.show()

    # Do some quick analysis to get start and end frame times for the video dataset to use
    # Find the time in the dataset where the rate of change is greater than 0.1
    start_frame = np.where(np.diff(us_pxdf) > 0.1)[0][0]
    print("Starting frame: ", start_frame)
    end_frame = np.where(np.diff(us_pxdf) > 0.1)[0][-1]
    print("Ending frame: ", end_frame)

    # Downsample the acceleration dataset to match the ultrasound dataset
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
    cross_corr = np.nan_to_num(cross_corr)    # Replace nan with 0

    # Compute the lag array
    lags = np.arange(-len(acc_mag_ds) + 1, len(us_pxdf)) # Compute lag array
    max_corr_index = np.argmax(cross_corr) # Find the index of the maximum cross-correlation value
    t_displacement = lags[max_corr_index] # Find the corresponding lag/time displacement
    t_shift = t_displacement / frame_rate
    print("Time displacement (lag) between signals (s):", t_shift)
    print("Max correlation value:", max(cross_corr))

    if show_cross_corr:
        # Plot the cross-correlation
        plt.figure()
        plt.title('Cross-Correlation')
        plt.plot(lags, cross_corr)
        plt.title('Cross-Correlation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        #plt.axvline(x=t_displacement, color='r', linestyle='--')
        plt.show()

    # Finally, now we can synchronize the 2 signals in terms of time. Adjust the time of the ultrasound signal and pad it
    # with zeros if needed.
    if t_shift > 0:
        us_pxdf = np.concatenate((np.zeros(int(t_shift * frame_rate)), us_pxdf))
    elif t_shift < 0:
        us_pxdf = us_pxdf[-int(t_shift * frame_rate):]

    t_us_aligned = np.arange(0, us_pxdf.shape[0]) / frame_rate
    print("New ultrasound dataset length with zero pad (s):", t_us_aligned[-1])

    if show_aligned_signals:
        # setting up figure and subplots
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax[0].plot(t_mocap, acc_mag, label='Acceleration Magnitude')
        ax[0].set_title('Acceleration Magnitude')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Acceleration (mm/s^2)')
        ax[0].legend()

        ax[1].plot(t_us_aligned, us_pxdf, label='Ultrasound Pixel Displacement')
        ax[1].set_title('Ultrasound Pixel Displacement')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Pixel Displacement')
        ax[1].legend()

        plt.tight_layout()
        plt.show()


    # Save the shift dataset to a file file
    with open(os.path.join(processed_data_folder, 'alignment_data.txt'), 'w') as f:
        f.write('Time displacement (lag) between signals (s): {}\n'.format(t_shift))
        f.write('Max Correlation Value: {}\n'.format(max(cross_corr)))
        f.write('Mocap Data Rate: {}\n'.format(mocap_rate))
        f.write('Ultrasound Video Length (s): {}\n'.format(video_length))
        f.write('Ultrasound Video Frame Rate: {}\n'.format(frame_rate))
        f.write('Start Frame: {}\n'.format(start_frame))
        f.write('End Frame: {}\n'.format(end_frame))



if __name__ == '__main__':

    print("This example script demonstrates how to align ultrasound and motion capture dataset.")

    # Path to the folder with the mocap dataset
    mocap_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\mocap\Session1'
    file_name = 'Recording002.trc'
    trc_path = os.path.join(mocap_path, file_name)

    # Path to the folder with the ultrasound dataset
    ultrasound_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\video'
    video_filename = 'raw002.mp4'
    video_path = os.path.join(ultrasound_path, video_filename)

    multimodal_alignment_example(trc_path, video_path,
                                 show_trc_animation=False,
                                 show_video=False,
                                 show_pixel_displacement=False,
                                 show_cross_corr=False,
                                 show_aligned_signals=False)



