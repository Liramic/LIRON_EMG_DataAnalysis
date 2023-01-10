from matplotlib import image as mpimg, pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from numpy.linalg import inv
import cv2  # for image loading+user electrode location selection, install 'opencv-python'
from General.HelperFunctions import GetSessions, getWeightsFileNameWithPath
import os

# electrode coordinates (empty if need to be obtained)
x_coor = []
y_coor = []


def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    global img
    img = cv2.imread(image_path, 1)  # for electrode location selection
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]
    # image resize
    # img = cv2.resize(img, (857, 1440))
    return image, height, width


def click_event(event, x, y, flags, params):
    global x_coor
    global y_coor
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        x_coor.append(x)
        y_coor.append(y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '(' + str(x) + ',' +
                    str(y) + ')', (x, y), font,
                    0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

    return x_coor, y_coor


def get_location():
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


def get_heatmap(image_path, number_of_channels, W, order):
    image, height, width = image_load(image_path)
    # load image

    get_location()

    # calculations for the heatmap
    inverse = np.absolute(inv(W))

    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]

    points = np.column_stack((x_coor, y_coor))

    f_interpolate = []
    for i in range(number_of_channels):
        f_interpolate.append(griddata(points, inverse[:, i], (grid_x, grid_y), method='linear'))

    # plot heatmap- don't delete
    fig, axs = plt.subplots(2, int(number_of_channels / 2), figsize=(16, 8))
    axs = axs.ravel()
    # plt.show the image
    for i in range(number_of_channels):
        axs[i].imshow(image)
        axs[i].pcolormesh(f_interpolate[order[i]], cmap='jet', alpha=0.5)
        axs[i].set_title("ICA Source %d" % (i + 1))
        axs[i].axis('off')

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()


if(__name__ == "__main__"):
    data_path = r"C:\Liron\DataEmg\Done"
    num_components=16
    order = np.arange(16)[::-1]
    #sessions = GetSessions(data_path)
    current_session = "04122022_1545"
    w = np.loadtxt(getWeightsFileNameWithPath(data_path, current_session), delimiter=",")
    get_heatmap(os.path.join(data_path, current_session, "A.jpeg"), num_components, w, order)