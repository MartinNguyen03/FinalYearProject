;; Auto-generated. Do not edit!


(when (boundp 'uni_lace_msgs::UniLaceStepService)
  (if (not (find-package "UNI_LACE_MSGS"))
    (make-package "UNI_LACE_MSGS"))
  (shadow 'UniLaceStepService (find-package "UNI_LACE_MSGS")))
(unless (find-package "UNI_LACE_MSGS::UNILACESTEPSERVICE")
  (make-package "UNI_LACE_MSGS::UNILACESTEPSERVICE"))
(unless (find-package "UNI_LACE_MSGS::UNILACESTEPSERVICEREQUEST")
  (make-package "UNI_LACE_MSGS::UNILACESTEPSERVICEREQUEST"))
(unless (find-package "UNI_LACE_MSGS::UNILACESTEPSERVICERESPONSE")
  (make-package "UNI_LACE_MSGS::UNILACESTEPSERVICERESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass uni_lace_msgs::UniLaceStepServiceRequest
  :super ros::object
  :slots (_act ))

(defmethod uni_lace_msgs::UniLaceStepServiceRequest
  (:init
   (&key
    ((:act __act) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _act __act)
   self)
  (:act
   (&rest __act)
   (if (keywordp (car __act))
       (send* _act __act)
     (progn
       (if __act (setq _act (car __act)))
       _act)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _act
    (send _act :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _act
       (send _act :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _act
     (send _act :deserialize buf ptr-) (incf ptr- (send _act :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UniLaceStepServiceResponse
  :super ros::object
  :slots (_obs _obs_raw ))

(defmethod uni_lace_msgs::UniLaceStepServiceResponse
  (:init
   (&key
    ((:obs __obs) (instance std_msgs::String :init))
    ((:obs_raw __obs_raw) (instance std_msgs::UInt8MultiArray :init))
    )
   (send-super :init)
   (setq _obs __obs)
   (setq _obs_raw __obs_raw)
   self)
  (:obs
   (&rest __obs)
   (if (keywordp (car __obs))
       (send* _obs __obs)
     (progn
       (if __obs (setq _obs (car __obs)))
       _obs)))
  (:obs_raw
   (&rest __obs_raw)
   (if (keywordp (car __obs_raw))
       (send* _obs_raw __obs_raw)
     (progn
       (if __obs_raw (setq _obs_raw (car __obs_raw)))
       _obs_raw)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _obs
    (send _obs :serialization-length)
    ;; std_msgs/UInt8MultiArray _obs_raw
    (send _obs_raw :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _obs
       (send _obs :serialize s)
     ;; std_msgs/UInt8MultiArray _obs_raw
       (send _obs_raw :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _obs
     (send _obs :deserialize buf ptr-) (incf ptr- (send _obs :serialization-length))
   ;; std_msgs/UInt8MultiArray _obs_raw
     (send _obs_raw :deserialize buf ptr-) (incf ptr- (send _obs_raw :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UniLaceStepService
  :super ros::object
  :slots ())

(setf (get uni_lace_msgs::UniLaceStepService :md5sum-) "af9f61a907e31117188b236c02aaf4bd")
(setf (get uni_lace_msgs::UniLaceStepService :datatype-) "uni_lace_msgs/UniLaceStepService")
(setf (get uni_lace_msgs::UniLaceStepService :request) uni_lace_msgs::UniLaceStepServiceRequest)
(setf (get uni_lace_msgs::UniLaceStepService :response) uni_lace_msgs::UniLaceStepServiceResponse)

(defmethod uni_lace_msgs::UniLaceStepServiceRequest
  (:response () (instance uni_lace_msgs::UniLaceStepServiceResponse :init)))

(setf (get uni_lace_msgs::UniLaceStepServiceRequest :md5sum-) "af9f61a907e31117188b236c02aaf4bd")
(setf (get uni_lace_msgs::UniLaceStepServiceRequest :datatype-) "uni_lace_msgs/UniLaceStepServiceRequest")
(setf (get uni_lace_msgs::UniLaceStepServiceRequest :definition-)
      "#request
std_msgs/String act

================================================================================
MSG: std_msgs/String
string data
---
#response
std_msgs/String obs
std_msgs/UInt8MultiArray obs_raw

================================================================================
MSG: std_msgs/String
string data

================================================================================
MSG: std_msgs/UInt8MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
uint8[]           data          # array of data


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

(setf (get uni_lace_msgs::UniLaceStepServiceResponse :md5sum-) "af9f61a907e31117188b236c02aaf4bd")
(setf (get uni_lace_msgs::UniLaceStepServiceResponse :datatype-) "uni_lace_msgs/UniLaceStepServiceResponse")
(setf (get uni_lace_msgs::UniLaceStepServiceResponse :definition-)
      "#request
std_msgs/String act

================================================================================
MSG: std_msgs/String
string data
---
#response
std_msgs/String obs
std_msgs/UInt8MultiArray obs_raw

================================================================================
MSG: std_msgs/String
string data

================================================================================
MSG: std_msgs/UInt8MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
uint8[]           data          # array of data


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



(provide :uni_lace_msgs/UniLaceStepService "af9f61a907e31117188b236c02aaf4bd")


