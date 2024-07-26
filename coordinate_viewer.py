import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def on_move(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        print(f"Coordinates: x={x:.2f}, y={y:.2f}")

def main(path_to_image):
    img = mpimg.imread(path_to_image)  # Replace with your image path
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()


if __name__ == "__main__":

    print("This is a really simple script to visualize an image to get the local coordinates of a point in the image.")
    print("This is more helpful for determining the scaling between the ultrasound image depth and pixel sizes")

    path_to_image = '/path/to/ultrasound-image/001-frame-extract.png'
    main(path_to_image)
