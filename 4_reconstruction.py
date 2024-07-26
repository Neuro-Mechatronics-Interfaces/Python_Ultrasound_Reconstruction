import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stlview import Object

import utils

def reconstruct(root_dir, show_animation=False, probe_model=None, probe_to_image_offset=(-50/2, -35/2)):

    # Load the ultrasound parameters
    us_params = np.load(os.path.join(root_dir, 'US_params.npz'))

    # Load freehand ultrasound images and masks
    print("Loading ultrasound images and masks...", end="")
    us_img_list = utils.natsort(glob.glob(os.path.join(root_dir, 'USdata', 'US', '*')))
    us_images = utils.loadImages(us_img_list, flip=True)
    us_mask_list = utils.natsort(glob.glob(os.path.join(root_dir, 'USdata', 'USmasks', '*')))
    us_masks = utils.loadImages(us_mask_list, flip=True)
    print("Done")

    # Load the alignment parameters
    align_params = np.load(os.path.join(root_dir, 'align_params.npz'))

    # Load the pose data and mocap parameters
    T_T_W = np.load(os.path.join(root_dir, 'target_pose.npy'))
    mocap_params = np.load(os.path.join(root_dir, 'mocap_params.npz'))

    # Calculate the new start and stop indices for the motion capture data
    us_duration = us_params['n_frames'] / us_params['frame_rate']  # Duration of the ultrasound video in seconds
    start_time = 0 - align_params['time_displacement']  # New start time for motion capture data
    stop_time = start_time + us_duration  # New stop time for motion capture data
    start_index = int(start_time * mocap_params['mocap_rate'])
    stop_index = int(stop_time * mocap_params['mocap_rate'])

    # Ensure indices are within bounds
    start_index = max(start_index, 0)
    stop_index = min(stop_index, len(T_T_W))

    # Extract the adjusted motion capture data
    T_T_W_adjusted = T_T_W[start_index:stop_index]

    # Downsample mocap data so its length matches the ultrasound frame count
    T_T_W_ds = utils.downsampleMatrixData(T_T_W_adjusted, us_params['n_frames'])

    # One more parse, index the ultrasound data and pose data by the start and end frame
    us_images_parsed = us_images[us_params['start_frame']:us_params['end_frame']]
    T_T_W_ds_parsed = T_T_W_ds[us_params['start_frame']:us_params['end_frame']]




    # The image offset from the probe tip center to the bottom image corner is (-50/2, 0, -35/2), so
    # we need to apply the offset and center the image too
    image_offset = np.array([probe_to_image_offset[0] + us_images_parsed[0].shape[1]/2,
                             us_images_parsed[0].shape[0],
                             probe_to_image_offset[1]])  # z and y seem to be switched

    # Set up figure and 3D axes for visualizing the ultrasound images and pose data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (frame, pose) in enumerate(zip(us_images_parsed, T_T_W_ds_parsed)):

        T = np.eye(4)
        T[:3, 3] = pose[:3, 3]
        T[:3, :3] = pose[:3, :3]

        # 3D map B-scans
        #X2, C2 = us3d.mapMask(imlist_us, masks_us, T_T_W, Params2)

        if show_animation:
            # We will draw the ultrasound image to match the pose data
            ax.clear()
            utils.plotImage(ax, frame, T)
            plt.draw()

            # Draw the ultrasound probe
            utils.drawModel(ax, pose[:3, :3], pose[:3, 3], probe_model, image_offset)

            # Set the plot limits
            t = pose[:3, 3]
            #ax.set_xlim([t[0] - 100, t[0] + 100])
            #ax.set_ylim([t[1] - 100, t[1] + 100])
            #ax.set_zlim([t[2] - 100, t[2] + 100])
            ax.set_xlim([300, 600])
            ax.set_ylim([400, 600])
            ax.set_zlim([-700, -500])

            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')

            plt.pause(0.1)

    print("done")

if __name__ == '__main__':

    print("This example reconstructs the 3D ultrasound volume from a sequence of 2D ultrasound images and pose data.")

    # The root directory needs to be where the dataset folder is located.
    root_dir = r'/Python_Ultrasound_Reconstruction/dataset'

    # We're using the probe model again
    probe_model_path = os.path.join(root_dir, '..', 'models', 'probe.stl')
    probe_model = Object(probe_model_path)
    probe_model.build()

    # Reconstruct the ultrasound volume
    reconstruct(root_dir, show_animation=True, probe_model=probe_model)
