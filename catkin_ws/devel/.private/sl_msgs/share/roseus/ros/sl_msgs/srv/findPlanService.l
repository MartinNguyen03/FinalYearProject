;; Auto-generated. Do not edit!


(when (boundp 'sl_msgs::findPlanService)
  (if (not (find-package "SL_MSGS"))
    (make-package "SL_MSGS"))
  (shadow 'findPlanService (find-package "SL_MSGS")))
(unless (find-package "SL_MSGS::FINDPLANSERVICE")
  (make-package "SL_MSGS::FINDPLANSERVICE"))
(unless (find-package "SL_MSGS::FINDPLANSERVICEREQUEST")
  (make-package "SL_MSGS::FINDPLANSERVICEREQUEST"))
(unless (find-package "SL_MSGS::FINDPLANSERVICERESPONSE")
  (make-package "SL_MSGS::FINDPLANSERVICERESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass sl_msgs::findPlanServiceRequest
  :super ros::object
  :slots (_pattern _aesthetic_mode ))

(defmethod sl_msgs::findPlanServiceRequest
  (:init
   (&key
    ((:pattern __pattern) (instance std_msgs::Int8MultiArray :init))
    ((:aesthetic_mode __aesthetic_mode) (instance std_msgs::Int8 :init))
    )
   (send-super :init)
   (setq _pattern __pattern)
   (setq _aesthetic_mode __aesthetic_mode)
   self)
  (:pattern
   (&rest __pattern)
   (if (keywordp (car __pattern))
       (send* _pattern __pattern)
     (progn
       (if __pattern (setq _pattern (car __pattern)))
       _pattern)))
  (:aesthetic_mode
   (&rest __aesthetic_mode)
   (if (keywordp (car __aesthetic_mode))
       (send* _aesthetic_mode __aesthetic_mode)
     (progn
       (if __aesthetic_mode (setq _aesthetic_mode (car __aesthetic_mode)))
       _aesthetic_mode)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8MultiArray _pattern
    (send _pattern :serialization-length)
    ;; std_msgs/Int8 _aesthetic_mode
    (send _aesthetic_mode :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8MultiArray _pattern
       (send _pattern :serialize s)
     ;; std_msgs/Int8 _aesthetic_mode
       (send _aesthetic_mode :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8MultiArray _pattern
     (send _pattern :deserialize buf ptr-) (incf ptr- (send _pattern :serialization-length))
   ;; std_msgs/Int8 _aesthetic_mode
     (send _aesthetic_mode :deserialize buf ptr-) (incf ptr- (send _aesthetic_mode :serialization-length))
   ;;
   self)
  )

(defclass sl_msgs::findPlanServiceResponse
  :super ros::object
  :slots (_plan ))

(defmethod sl_msgs::findPlanServiceResponse
  (:init
   (&key
    ((:plan __plan) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _plan __plan)
   self)
  (:plan
   (&rest __plan)
   (if (keywordp (car __plan))
       (send* _plan __plan)
     (progn
       (if __plan (setq _plan (car __plan)))
       _plan)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _plan
    (send _plan :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _plan
       (send _plan :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _plan
     (send _plan :deserialize buf ptr-) (incf ptr- (send _plan :serialization-length))
   ;;
   self)
  )

(defclass sl_msgs::findPlanService
  :super ros::object
  :slots ())

(setf (get sl_msgs::findPlanService :md5sum-) "e752f9c85170cb73cc8cf790042cfd2d")
(setf (get sl_msgs::findPlanService :datatype-) "sl_msgs/findPlanService")
(setf (get sl_msgs::findPlanService :request) sl_msgs::findPlanServiceRequest)
(setf (get sl_msgs::findPlanService :response) sl_msgs::findPlanServiceResponse)

(defmethod sl_msgs::findPlanServiceRequest
  (:response () (instance sl_msgs::findPlanServiceResponse :init)))

(setf (get sl_msgs::findPlanServiceRequest :md5sum-) "e752f9c85170cb73cc8cf790042cfd2d")
(setf (get sl_msgs::findPlanServiceRequest :datatype-) "sl_msgs/findPlanServiceRequest")
(setf (get sl_msgs::findPlanServiceRequest :definition-)
      "std_msgs/Int8MultiArray pattern
std_msgs/Int8 aesthetic_mode

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
uint32 stride  # stride of given dimension
================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/String plan

================================================================================
MSG: std_msgs/String
string data
")

(setf (get sl_msgs::findPlanServiceResponse :md5sum-) "e752f9c85170cb73cc8cf790042cfd2d")
(setf (get sl_msgs::findPlanServiceResponse :datatype-) "sl_msgs/findPlanServiceResponse")
(setf (get sl_msgs::findPlanServiceResponse :definition-)
      "std_msgs/Int8MultiArray pattern
std_msgs/Int8 aesthetic_mode

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
uint32 stride  # stride of given dimension
================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/String plan

================================================================================
MSG: std_msgs/String
string data
")



(provide :sl_msgs/findPlanService "e752f9c85170cb73cc8cf790042cfd2d")


