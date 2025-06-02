import cv2
import rospy
import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image


pil_mode_channels = {'L' : 1, 'RGB' : 3, 'RGBA' : 4, 'YCbCr' : 3 }
encoding_to_pil_mode={
    '16UC1':        'L',
    'bayer_grbg8':  'L',
    'mono8':        'L',
    '8UC1':         'L',
    '8UC3':         'RGB',
    'rgb8':         'RGB',
    'bgr8':         'RGB',
    'rgba8':        'RGBA',
    'bgra8':        'RGBA',
    'bayer_rggb':   'L',
    'bayer_gbrg':   'L',
    'bayer_grbg':   'L',
    'bayer_bggr':   'L',
    'yuv422':       'YCbCr',
    'yuv411':       'YCbCr'}

""" transforms a sensor_msgs/Image message to a numpy array """
def msg_to_img(msg, encoding='bgr'):
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, -1)
    if encoding=='bgr':
        return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    else:
        return img

""" transforms a sensor_msgs/Image message to a PIL image """
def msg_to_pil(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, -1)
    img = PILImage.fromarray(img, encoding_to_pil_mode[msg.encoding])
    return img

""" transforms a numpy array image to a PIL image """
def img_to_pil(img, encoding='bgr8'):
    return PILImage.fromarray(img, encoding_to_pil_mode[encoding])

""" transforms a numpy array into a sensor_msgs/Image message """
def img_to_msg(img, encoding="bgr8"):
    # height, width = np.size(img)[:2] # H,W,C
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.height = np.shape(img)[0]
    msg.width = np.shape(img)[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(np.size(img)/np.shape(img)[0])
    msg.data = img.tobytes()
    return msg

""" transforms a PIL Image into a sensor_msgs/Image message """
def pil_to_msg(img, encoding="rgb8"):
    # height, width = np.size(img)[:2] # H,W,C
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.height = img.height
    msg.width = img.width
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = pil_mode_channels[encoding_to_pil_mode[encoding]]*img.width
    msg.data = img.tobytes()
    return msg

""" transforms a PIL Image into a numpy array image """
def pil_to_img(img):
    return np.array(img)

""" rescale a numpy array image to a given percentage """
def rescale(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

""" zoom into a numpy array image by a given scale """
def zoom(img, scale, centre=None):
    if not centre:  
        centre = [img.shape[0]//2, img.shape[1]//2] # y,x
    half_h = int(img.shape[0]/2 * scale)
    half_w = int(img.shape[1]/2 * scale)
    img =img[centre[0]-half_h:centre[0]+half_h, centre[1]-half_w:centre[1]+half_w]
    return img, [centre[0]-half_h, centre[1]-half_w]

""" concatenate two PIL images together """
def concat_pil(im1, im2):
    dst = PILImage.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst