import cv2
import numpy as np

class ColourSegmentation:
    """
    ColourSegmentation allows segmentation of regions in an image based on RGB colour thresholds.
    It supports interactive threshold adjustment using OpenCV trackbars and applies morphological
    operations (e.g. opening, closing) to clean up the resulting mask.
    
    Morphology Operations Mapping:
    0: Erosion
    1: Dilation
    2: Opening (Erosion followed by Dilation)
    3: Closing (Dilation followed by Erosion)
    """

    # Class-level variable to hold threshold pairs [[lower RGB], [upper RGB]]
    thresholds = np.array([[[0, 0, 0], [0, 0, 0]]])

    def __init__(self, thresh_l, thresh_h, image_grabber, kernel_size=3, morph_op=[2, 3], live_adjust=True):
        """
        Initialise the ColourSegmentation class.

        Args:
            thresh_l (list[int]): Lower RGB threshold (e.g. [B, G, R]).
            thresh_h (list[int]): Upper RGB threshold.
            image_grabber (Callable): Function to grab the current image (frame).
            kernel_size (int): Size of the structuring element for morphological operations.
            morph_op (list[int]): List of morphology operation codes (0 to 3).
            live_adjust (bool): If True, show GUI sliders to adjust thresholds in real-time.
        """
        self.thresh_l = thresh_l
        self.thresh_h = thresh_h
        self.update_thresholds(thresh_l, thresh_h)  # Initialise threshold array
        self.image_grabber = image_grabber          # Function to grab image input
        self.kernel_size = kernel_size              # Kernel size for morphological operations
        self.morphology_operation = morph_op        # List of morphology operations to apply

        # If live adjustment is enabled, start the interactive GUI
        if live_adjust:
            self.set_thresholds()

    def set_thresholds(self):
        """
        Displays OpenCV trackbars for adjusting RGB thresholds live. The updated thresholds
        are applied to the image to generate and show the binary mask in real-time.
        """
        cv2.namedWindow('Mask')  # Create window for trackbars and mask display

        # Create trackbars for each RGB channel (lower and upper bounds)
        cv2.createTrackbar('B LOW :', 'Mask', self.thresh_l[0], 255, self.nothing)
        cv2.createTrackbar('B HIGH:', 'Mask', self.thresh_h[0], 255, self.nothing)
        cv2.createTrackbar('G LOW :', 'Mask', self.thresh_l[1], 255, self.nothing)
        cv2.createTrackbar('G HIGH:', 'Mask', self.thresh_h[1], 255, self.nothing)
        cv2.createTrackbar('R LOW :', 'Mask', self.thresh_l[2], 255, self.nothing)
        cv2.createTrackbar('R HIGH:', 'Mask', self.thresh_h[2], 255, self.nothing)

        # Continuously update thresholds and display mask until ESC key is pressed
        while True:
            img = self.image_grabber()  # Get current image/frame

            # Read current positions of trackbars
            self.thresh_l[0] = cv2.getTrackbarPos('B LOW :', 'Mask')
            self.thresh_h[0] = cv2.getTrackbarPos('B HIGH:', 'Mask')
            self.thresh_l[1] = cv2.getTrackbarPos('G LOW :', 'Mask')
            self.thresh_h[1] = cv2.getTrackbarPos('G HIGH:', 'Mask')
            self.thresh_l[2] = cv2.getTrackbarPos('R LOW :', 'Mask')
            self.thresh_h[2] = cv2.getTrackbarPos('R HIGH:', 'Mask')

            # Update internal threshold representation
            self.update_thresholds(self.thresh_l, self.thresh_h)

            # Predict mask using current thresholds
            mask = self.predict_img(img)

            # Show rescaled mask and original image side-by-side
            cv2.imshow("Mask", cv2.resize(mask, (350, int(mask.shape[0]/mask.shape[1]*350)), interpolation=cv2.INTER_AREA))
            cv2.imshow("Original", img)

            # Exit if ESC key is pressed
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break

    @staticmethod
    def nothing(val):
        """
        Dummy callback function required by OpenCV trackbars.
        """
        pass

    def update_thresholds(self, thresh_l, thresh_h):
        """
        Updates the internal threshold array used for segmentation.
        Handles wrap-around colour thresholds (i.e., low > high).

        Args:
            thresh_l (list[int]): Lower HSV bounds.
            thresh_h (list[int]): Upper HSV bounds.
        """
        thresholds = np.array([[[0, 0, 0], [0, 0, 0]]])

        # Iterate through each channel: B=0, G=1, R=2
        for id, (high, low) in enumerate(zip(thresh_h, thresh_l)):
            if high > low:
                # Normal threshold range (low < high)
                for t in thresholds:
                    t[0, id] = low
                    t[1, id] = high
            else:
                # Wrap-around: e.g. low=200, high=50
                thresholds = np.repeat(thresholds, 2, axis=0)  # Duplicate for wrap
                for j, t in enumerate(thresholds):
                    if j % 2 == 0:
                        t[0, id] = low
                        t[1, id] = 255
                    else:
                        t[0, id] = 0
                        t[1, id] = high

        # Save updated threshold array
        self.thresholds = thresholds

    def predict_img(self, img):
        """
        Generate a binary mask from the given image using the stored colour thresholds.
        Applies morphological operations to clean the mask.

        Args:
            img (np.ndarray): BGR image to process.

        Returns:
            np.ndarray: Binary mask where white pixels represent segmented regions.
        """
        # Initialise blank mask of the same height and width
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Apply all threshold ranges (ORed together)
        for t in self.thresholds:
            mask = cv2.bitwise_or(mask, cv2.inRange(img, t[0], t[1]))

        # Define structuring element (cross-shaped kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self.kernel_size, self.kernel_size))

        # Apply selected morphological operations in sequence
        for op in self.morphology_operation:
            if op == 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            elif op == 1:
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            elif op == 2:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            elif op == 3:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            else:
                print("Unknown Morphology Operation!")

        return mask

    def colourSegmentation(self, img, lower, upper):
        """
        Perform simple colour-based segmentation using provided bounds.

        Args:
            img (np.ndarray): Input image in BGR format.
            lower (list[int]): Lower bound for BGR values.
            upper (list[int]): Upper bound for BGR values.

        Returns:
            np.ndarray: Binary mask with white pixels where colours are in range.
        """
        return cv2.inRange(img, lower, upper)
