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

    path_to_image = '001-frame-extract.png'
    main(path_to_image)
