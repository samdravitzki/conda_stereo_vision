import cv2
import numpy as np
import glob
from PIL import ExifTags
from PIL import Image
from matplotlib import pyplot
import numpy.linalg as la


class Calibrator:

    def __init__(self, row_count, col_count):
        self.row_count = row_count
        self.col_count = col_count

        self.ret = None
        self.camera_matrix = None
        self.distortion_coeff = None
        self.vecs = None
        self.focal_length = None

    def calibrate(self, input_directory, output_directory):
        chessboard = (self.row_count, self.col_count)

        objpoints = []
        imgpoints = []
        # Prepare grid and the points to display
        objp = np.zeros((np.product(chessboard), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

        calibration_paths = glob.glob(input_directory)

        for i, image_path in enumerate(calibration_paths):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Image loaded, Analysing {0}...".format(image_path))
            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard, None)

            if ret == True:
                print("Chessboard detected!")
                # define criteria for subpixel accuracy
                critera = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                # refine corner location based on criteria
                improved_corners = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), critera)
                objpoints.append(objp)
                imgpoints.append(improved_corners)
                image = cv2.drawChessboardCorners(image, chessboard, improved_corners, ret)
                cv2.imwrite("{0}draw-{1}.jpg".format(output_directory, i), image)
            else:
                print("Chessboard not detected!")
        # calibrate image based on the images
        ret, camera_matrix, distortion_coeff, rvecs, tvecs \
            = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

        self.ret = ret
        self.camera_matrix = camera_matrix
        self.distortion_coeff = distortion_coeff
        self.vecs = (rvecs, tvecs)

        # Get the focal-point form the mobile image meta data

        exif_image = Image.open(calibration_paths[0])

        # Select all meta data from the first image
        exif_data = {
            ExifTags.TAGS[k]: v
            for k, v in exif_image._getexif().items()
            if k in ExifTags.TAGS
        }

        # Select the focal length
        focal_length_exif = exif_data['FocalLength']

        self.focal_length = focal_length_exif[0] / focal_length_exif[1]
        self.save_parameters()

    def undistort_image(self, image):
        # Get the height and width (images must have equal size)
        image_height, image_width = image.shape[:2]

        # Calculate an optimal matrix based on the free scaling parameter
        # (A parameter that allows us to rescale the image and still keep a valid matrix)
        optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion_coeff,
            (image_width, image_height), 1, (image_width, image_height))

        # Undistort the images with the optimal camera matrix
        image_undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.distortion_coeff,
            newCameraMatrix=optimal_camera_matrix)

        # pyplot.imshow(image_undistorted, 'gray')
        # pyplot.show()

        x, y, w, h = roi
        image_undistorted = image_undistorted[y:y + h, x:x + w]

        # pyplot.imshow(image_undistorted, 'gray')
        # pyplot.show()

        return image_undistorted

    def save_parameters(self):
        rvecs, tvecs = self.vecs
        np.save("./camera_parameters/ret", self.ret)
        np.save("./camera_parameters/K", self.camera_matrix)
        np.save("./camera_parameters/dist", self.distortion_coeff)
        np.save("./camera_parameters/rvecs", rvecs)
        np.save("./camera_parameters/tvecs", tvecs)
        np.save("./camera_parameters/focal_length", self.focal_length)

    def load_parameters(self):
        self.ret = np.load("./camera_parameters/ret.npy")
        self.camera_matrix = np.load("./camera_parameters/K.npy")
        self.distortion_coeff = np.load("./camera_parameters/dist.npy")
        rvecs = np.load("./camera_parameters/rvecs.npy")
        tvecs = np.load("./camera_parameters/tvecs.npy")
        self.vecs = (rvecs, tvecs)
        self.focal_length = np.load("./camera_parameters/focal_length.npy")


