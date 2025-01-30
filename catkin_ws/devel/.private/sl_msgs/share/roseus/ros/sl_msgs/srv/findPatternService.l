;; Auto-generated. Do not edit!


(when (boundp 'sl_msgs::findPatternService)
  (if (not (find-package "SL_MSGS"))
    (make-package "SL_MSGS"))
  (shadow 'findPatternService (find-package "SL_MSGS")))
(unless (find-package "SL_MSGS::FINDPATTERNSERVICE")
  (make-package "SL_MSGS::FINDPATTERNSERVICE"))
(unless (find-package "SL_MSGS::FINDPATTERNSERVICEREQUEST")
  (make-package "SL_MSGS::FINDPATTERNSERVICEREQUEST"))
(unless (find-package "SL_MSGS::FINDPATTERNSERVICERESPONSE")
  (make-package "SL_MSGS::FINDPATTERNSERVICERESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass sl_msgs::findPatternServiceRequest
  :super ros::object
  :slots (_num_eyelets _shoelace_length _eyelet_positions ))

(defmethod sl_msgs::findPatternServiceRequest
  (:init
   (&key
    ((:num_eyelets __num_eyelets) (instance std_msgs::Int8 :init))
    ((:shoelace_length __shoelace_length) (instance std_msgs::Float32 :init))
    ((:eyelet_positions __eyelet_positions) (instance std_msgs::Float32MultiArray :init))
    )
   (send-super :init)
   (setq _num_eyelets __num_eyelets)
   (setq _shoelace_length __shoelace_length)
   (setq _eyelet_positions __eyelet_positions)
   self)
  (:num_eyelets
   (&rest __num_eyelets)
   (if (keywordp (car __num_eyelets))
       (send* _num_eyelets __num_eyelets)
     (progn
       (if __num_eyelets (setq _num_eyelets (car __num_eyelets)))
       _num_eyelets)))
  (:shoelace_length
   (&rest __shoelace_length)
   (if (keywordp (car __shoelace_length))
       (send* _shoelace_length __shoelace_length)
     (progn
       (if __shoelace_length (setq _shoelace_length (car __shoelace_length)))
       _shoelace_length)))
  (:eyelet_positions
   (&rest __eyelet_positions)
   (if (keywordp (car __eyelet_positions))
       (send* _eyelet_positions __eyelet_positions)
     (progn
       (if __eyelet_positions (setq _eyelet_positions (car __eyelet_positions)))
       _eyelet_positions)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _num_eyelets
    (send _num_eyelets :serialization-length)
    ;; std_msgs/Float32 _shoelace_length
    (send _shoelace_length :serialization-length)
    ;; std_msgs/Float32MultiArray _eyelet_positions
    (send _eyelet_positions :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _num_eyelets
       (send _num_eyelets :serialize s)
     ;; std_msgs/Float32 _shoelace_length
       (send _shoelace_length :serialize s)
     ;; std_msgs/Float32MultiArray _eyelet_positions
       (send _eyelet_positions :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _num_eyelets
     (send _num_eyelets :deserialize buf ptr-) (incf ptr- (send _num_eyelets :serialization-length))
   ;; std_msgs/Float32 _shoelace_length
     (send _shoelace_length :deserialize buf ptr-) (incf ptr- (send _shoelace_length :serialization-length))
   ;; std_msgs/Float32MultiArray _eyelet_positions
     (send _eyelet_positions :deserialize buf ptr-) (incf ptr- (send _eyelet_positions :serialization-length))
   ;;
   self)
  )

(defclass sl_msgs::findPatternServiceResponse
  :super ros::object
  :slots (_pattern ))

(defmethod sl_msgs::findPatternServiceResponse
  (:init
   (&key
    ((:pattern __pattern) (instance std_msgs::Int8MultiArray :init))
    )
   (send-super :init)
   (setq _pattern __pattern)
   self)
  (:pattern
   (&rest __pattern)
   (if (keywordp (car __pattern))
       (send* _pattern __pattern)
     (progn
       (if __pattern (setq _pattern (car __pattern)))
       _pattern)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8MultiArray _pattern
    (send _pattern :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8MultiArray _pattern
       (send _pattern :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8MultiArray _pattern
     (send _pattern :deserialize buf ptr-) (incf ptr- (send _pattern :serialization-length))
   ;;
   self)
  )

(defclass sl_msgs::findPatternService
  :super ros::object
  :slots ())

(setf (get sl_msgs::findPatternService :md5sum-) "145ffa40f05356208146cb6e34948f0c")
(setf (get sl_msgs::findPatternService :datatype-) "sl_msgs/findPatternService")
(setf (get sl_msgs::findPatternService :request) sl_msgs::findPatternServiceRequest)
(setf (get sl_msgs::findPatternService :response) sl_msgs::findPatternServiceResponse)

(defmethod sl_msgs::findPatternServiceRequest
  (:response () (instance sl_msgs::findPatternServiceResponse :init)))

(setf (get sl_msgs::findPatternServiceRequest :md5sum-) "145ffa40f05356208146cb6e34948f0c")
(setf (get sl_msgs::findPatternServiceRequest :datatype-) "sl_msgs/findPatternServiceRequest")
(setf (get sl_msgs::findPatternServiceRequest :definition-)
      "std_msgs/Int8 num_eyelets
std_msgs/Float32 shoelace_length
std_msgs/Float32MultiArray eyelet_positions

================================================================================
MSG: std_msgs/Int8
int8 data

================================================================================
MSG: std_msgs/Float32
float32 data
================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


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
---
std_msgs/Int8MultiArray pattern

================================================================================
MSG: std_msgs/Int8MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
int8[]            data          # array of data


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

(setf (get sl_msgs::findPatternServiceResponse :md5sum-) "145ffa40f05356208146cb6e34948f0c")
(setf (get sl_msgs::findPatternServiceResponse :datatype-) "sl_msgs/findPatternServiceResponse")
(setf (get sl_msgs::findPatternServiceResponse :definition-)
      "std_msgs/Int8 num_eyelets
std_msgs/Float32 shoelace_length
std_msgs/Float32MultiArray eyelet_positions

================================================================================
MSG: std_msgs/Int8
int8 data

================================================================================
MSG: std_msgs/Float32
float32 data
================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


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
---
std_msgs/Int8MultiArray pattern

================================================================================
MSG: std_msgs/Int8MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
int8[]            data          # array of data


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



(provide :sl_msgs/findPatternService "145ffa40f05356208146cb6e34948f0c")


