import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import utils


def draw_plots(us_px_diff, acc_mag_ds, us_params, lags, cross_corr, t_displacement):
    """ Helper function for plotting the signals and cross-correlation
    """
    t_us = np.arange(len(us_px_diff)) / us_params['frame_rate']
    print("video length: ", us_params['n_frames'] / us_params['frame_rate'])

    # Trim length of the pose data to match the ultrasound video length
    t_d_s = t_displacement / us_params['frame_rate']
    if t_d_s > 0:
        acc_mag_ds = acc_mag_ds[:len(us_px_diff)]
    else:
        us_px_diff = us_px_diff[:len(acc_mag_ds)]
        t_us = t_us[:len(acc_mag_ds)]

    # Create Subplots for correlation, acceleration and pixel displacement
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    # Plot the correlation, index iw tih a line, and show the time displacement in seconds
    axs[0].plot(lags, cross_corr, label='Correlation')
    axs[0].axvline(x=t_displacement, color='r', linestyle='--', label='Max Correlation')
    axs[0].text(t_displacement, max(cross_corr)*0.8, f"lag (s): {t_d_s}", color='r')
    axs[0].set_ylabel('Cross-correlation')
    axs[0].set_xlabel('Lag')
    axs[0].set_title('Cross-correlation between signals')
    axs[0].legend()

    axs[1].plot(t_us, acc_mag_ds, label='Acceleration Magnitude')
    axs[1].set_ylabel('Acceleration Magnitude')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Acceleration Magnitude')
    axs[1].legend()

    axs[2].plot(t_us, us_px_diff, label='Pixel Displacement')
    axs[2].set_ylabel('Pixel Displacement')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('Pixel Displacement')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def align_us_and_mocap(root_dir, show_video=False, show_corr=False):
    # ============================= Ultrasound data processing =============================
    # Load the ultrasound parameters
    us_params = np.load(os.path.join(root_dir, 'us_params.npz'))

    # Prep image paths from freehand ultrasound
    us_img_list = utils.natsort(glob.glob(os.path.join(root_dir, 'USdata', 'US', '*')))
    print("Total number of ultrasound images: ", len(us_img_list))

    # Get the 1D pixel displacement between frames
    us_px_diff = utils.getPixelDisplacement(us_img_list, show_video)

    # ============================= Motion capture data processing =============================
    # Load the pose data
    T_T_W = np.load(os.path.join(root_dir, 'target_pose.npy'))

    # Get the acceleration magnitude from the pose data
    acc_mag = utils.computeAccelerationMagnitude(T_T_W)

    # Downsample data to match ultrasound frame rate
    acc_mag_ds = utils.downsampleByAveraging(acc_mag, len(us_px_diff))

    # Normalize the acceleration magnitude signal
    acc_mag_ds = acc_mag_ds / np.max(acc_mag_ds)

    # Zero out the first and last 3 seconds of the signal, avoid signal artifacts from jostling the probe
    zero_mask = int(us_params['frame_rate'] * 3)
    acc_mag_ds[:zero_mask] = 0.0
    acc_mag_ds[-zero_mask:] = 0.0

    # ===================================== Synchronization ====================================
    # Perform cross-correlation to find the time shift between the 2 signals
    cross_corr = np.correlate(us_px_diff, acc_mag_ds, mode='full')
    cross_corr = np.nan_to_num(cross_corr)    # Replace nan with 0

    # Compute the lag array
    lags = np.arange(-len(acc_mag_ds) + 1, len(us_px_diff)) # Compute lag array
    max_corr_index = np.argmax(cross_corr)  # Find the index of the maximum cross-correlation value
    t_displacement = lags[max_corr_index]  # Find the corresponding lag/time displacement
    t_shift = t_displacement / us_params['frame_rate']
    print("Time displacement (lag) between signals (s):", t_shift)
    print("Max correlation value:", max(cross_corr))

    # Visualize the signal comparison and correlation
    if show_corr:
        draw_plots(us_px_diff, acc_mag_ds, us_params, lags, cross_corr, t_displacement)

    # Create alignment params dict to save
    align_params = {
        'time_displacement': t_shift,
        'max_corr_index': max(cross_corr),
        'max_corr_val': max(cross_corr)
    }
    return align_params


if __name__ == '__main__':

    print("This example demonstrates how to align the ultrasound and motion capture dataset. Assumes "
          "'target_pose.npy' exists and ultrasound video frames are saved in the dataset folder.")

    # The root directory needs to be where the dataset folder is located.
    root_dir = r'/Python_Ultrasound_Reconstruction/dataset'

    align_params = align_us_and_mocap(root_dir, show_video=False, show_corr=False)

    # Save the time lag parameters to a .npz file
    np.savez(os.path.join(root_dir, 'align_params.npz'), **align_params)
