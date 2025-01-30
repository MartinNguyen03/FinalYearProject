;; Auto-generated. Do not edit!


(when (boundp 'sl_msgs::findTargetsService)
  (if (not (find-package "SL_MSGS"))
    (make-package "SL_MSGS"))
  (shadow 'findTargetsService (find-package "SL_MSGS")))
(unless (find-package "SL_MSGS::FINDTARGETSSERVICE")
  (make-package "SL_MSGS::FINDTARGETSSERVICE"))
(unless (find-package "SL_MSGS::FINDTARGETSSERVICEREQUEST")
  (make-package "SL_MSGS::FINDTARGETSSERVICEREQUEST"))
(unless (find-package "SL_MSGS::FINDTARGETSSERVICERESPONSE")
  (make-package "SL_MSGS::FINDTARGETSSERVICERESPONSE"))

(in-package "ROS")



(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass sl_msgs::findTargetsServiceRequest
  :super ros::object
  :slots (_target_name _camera_name ))

(defmethod sl_msgs::findTargetsServiceRequest
  (:init
   (&key
    ((:target_name __target_name) "")
    ((:camera_name __camera_name) "")
    )
   (send-super :init)
   (setq _target_name (string __target_name))
   (setq _camera_name (string __camera_name))
   self)
  (:target_name
   (&optional __target_name)
   (if __target_name (setq _target_name __target_name)) _target_name)
  (:camera_name
   (&optional __camera_name)
   (if __camera_name (setq _camera_name __camera_name)) _camera_name)
  (:serialization-length
   ()
   (+
    ;; string _target_name
    4 (length _target_name)
    ;; string _camera_name
    4 (length _camera_name)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _target_name
       (write-long (length _target_name) s) (princ _target_name s)
     ;; string _camera_name
       (write-long (length _camera_name) s) (princ _camera_name s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _target_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _target_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _camera_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _camera_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass sl_msgs::findTargetsServiceResponse
  :super ros::object
  :slots (_target _confidence ))

(defmethod sl_msgs::findTargetsServiceResponse
  (:init
   (&key
    ((:target __target) (instance geometry_msgs::PoseArray :init))
    ((:confidence __confidence) (instance std_msgs::Float64MultiArray :init))
    )
   (send-super :init)
   (setq _target __target)
   (setq _confidence __confidence)
   self)
  (:target
   (&rest __target)
   (if (keywordp (car __target))
       (send* _target __target)
     (progn
       (if __target (setq _target (car __target)))
       _target)))
  (:confidence
   (&rest __confidence)
   (if (keywordp (car __confidence))
       (send* _confidence __confidence)
     (progn
       (if __confidence (setq _confidence (car __confidence)))
       _confidence)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/PoseArray _target
    (send _target :serialization-length)
    ;; std_msgs/Float64MultiArray _confidence
    (send _confidence :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/PoseArray _target
       (send _target :serialize s)
     ;; std_msgs/Float64MultiArray _confidence
       (send _confidence :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/PoseArray _target
     (send _target :deserialize buf ptr-) (incf ptr- (send _target :serialization-length))
   ;; std_msgs/Float64MultiArray _confidence
     (send _confidence :deserialize buf ptr-) (incf ptr- (send _confidence :serialization-length))
   ;;
   self)
  )

(defclass sl_msgs::findTargetsService
  :super ros::object
  :slots ())

(setf (get sl_msgs::findTargetsService :md5sum-) "aa823e51a177fa5a87a4dc28394b4618")
(setf (get sl_msgs::findTargetsService :datatype-) "sl_msgs/findTargetsService")
(setf (get sl_msgs::findTargetsService :request) sl_msgs::findTargetsServiceRequest)
(setf (get sl_msgs::findTargetsService :response) sl_msgs::findTargetsServiceResponse)

(defmethod sl_msgs::findTargetsServiceRequest
  (:response () (instance sl_msgs::findTargetsServiceResponse :init)))

(setf (get sl_msgs::findTargetsServiceRequest :md5sum-) "aa823e51a177fa5a87a4dc28394b4618")
(setf (get sl_msgs::findTargetsServiceRequest :datatype-) "sl_msgs/findTargetsServiceRequest")
(setf (get sl_msgs::findTargetsServiceRequest :definition-)
      "string target_name
string camera_name
---
geometry_msgs/PoseArray target
std_msgs/Float64MultiArray confidence

================================================================================
MSG: geometry_msgs/PoseArray
# An array of poses with a header for global reference.

Header header

Pose[] poses

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: std_msgs/Float64MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float64[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")

(setf (get sl_msgs::findTargetsServiceResponse :md5sum-) "aa823e51a177fa5a87a4dc28394b4618")
(setf (get sl_msgs::findTargetsServiceResponse :datatype-) "sl_msgs/findTargetsServiceResponse")
(setf (get sl_msgs::findTargetsServiceResponse :definition-)
      "string target_name
string camera_name
---
geometry_msgs/PoseArray target
std_msgs/Float64MultiArray confidence

================================================================================
MSG: geometry_msgs/PoseArray
# An array of poses with a header for global reference.

Header header

Pose[] poses

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: std_msgs/Float64MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float64[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")



(provide :sl_msgs/findTargetsService "aa823e51a177fa5a87a4dc28394b4618")


