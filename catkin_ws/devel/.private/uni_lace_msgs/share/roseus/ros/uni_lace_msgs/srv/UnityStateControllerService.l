;; Auto-generated. Do not edit!


(when (boundp 'uni_lace_msgs::UnityStateControllerService)
  (if (not (find-package "UNI_LACE_MSGS"))
    (make-package "UNI_LACE_MSGS"))
  (shadow 'UnityStateControllerService (find-package "UNI_LACE_MSGS")))
(unless (find-package "UNI_LACE_MSGS::UNITYSTATECONTROLLERSERVICE")
  (make-package "UNI_LACE_MSGS::UNITYSTATECONTROLLERSERVICE"))
(unless (find-package "UNI_LACE_MSGS::UNITYSTATECONTROLLERSERVICEREQUEST")
  (make-package "UNI_LACE_MSGS::UNITYSTATECONTROLLERSERVICEREQUEST"))
(unless (find-package "UNI_LACE_MSGS::UNITYSTATECONTROLLERSERVICERESPONSE")
  (make-package "UNI_LACE_MSGS::UNITYSTATECONTROLLERSERVICERESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass uni_lace_msgs::UnityStateControllerServiceRequest
  :super ros::object
  :slots (_real_pos ))

(defmethod uni_lace_msgs::UnityStateControllerServiceRequest
  (:init
   (&key
    ((:real_pos __real_pos) (instance geometry_msgs::PoseArray :init))
    )
   (send-super :init)
   (setq _real_pos __real_pos)
   self)
  (:real_pos
   (&rest __real_pos)
   (if (keywordp (car __real_pos))
       (send* _real_pos __real_pos)
     (progn
       (if __real_pos (setq _real_pos (car __real_pos)))
       _real_pos)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/PoseArray _real_pos
    (send _real_pos :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/PoseArray _real_pos
       (send _real_pos :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/PoseArray _real_pos
     (send _real_pos :deserialize buf ptr-) (incf ptr- (send _real_pos :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UnityStateControllerServiceResponse
  :super ros::object
  :slots (_sim_pos ))

(defmethod uni_lace_msgs::UnityStateControllerServiceResponse
  (:init
   (&key
    ((:sim_pos __sim_pos) (instance geometry_msgs::PoseArray :init))
    )
   (send-super :init)
   (setq _sim_pos __sim_pos)
   self)
  (:sim_pos
   (&rest __sim_pos)
   (if (keywordp (car __sim_pos))
       (send* _sim_pos __sim_pos)
     (progn
       (if __sim_pos (setq _sim_pos (car __sim_pos)))
       _sim_pos)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/PoseArray _sim_pos
    (send _sim_pos :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/PoseArray _sim_pos
       (send _sim_pos :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/PoseArray _sim_pos
     (send _sim_pos :deserialize buf ptr-) (incf ptr- (send _sim_pos :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UnityStateControllerService
  :super ros::object
  :slots ())

(setf (get uni_lace_msgs::UnityStateControllerService :md5sum-) "f41682cf3f385f5d1be3d8588334a602")
(setf (get uni_lace_msgs::UnityStateControllerService :datatype-) "uni_lace_msgs/UnityStateControllerService")
(setf (get uni_lace_msgs::UnityStateControllerService :request) uni_lace_msgs::UnityStateControllerServiceRequest)
(setf (get uni_lace_msgs::UnityStateControllerService :response) uni_lace_msgs::UnityStateControllerServiceResponse)

(defmethod uni_lace_msgs::UnityStateControllerServiceRequest
  (:response () (instance uni_lace_msgs::UnityStateControllerServiceResponse :init)))

(setf (get uni_lace_msgs::UnityStateControllerServiceRequest :md5sum-) "f41682cf3f385f5d1be3d8588334a602")
(setf (get uni_lace_msgs::UnityStateControllerServiceRequest :datatype-) "uni_lace_msgs/UnityStateControllerServiceRequest")
(setf (get uni_lace_msgs::UnityStateControllerServiceRequest :definition-)
      "#request
geometry_msgs/PoseArray real_pos

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
---
#response
geometry_msgs/PoseArray sim_pos

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
")

(setf (get uni_lace_msgs::UnityStateControllerServiceResponse :md5sum-) "f41682cf3f385f5d1be3d8588334a602")
(setf (get uni_lace_msgs::UnityStateControllerServiceResponse :datatype-) "uni_lace_msgs/UnityStateControllerServiceResponse")
(setf (get uni_lace_msgs::UnityStateControllerServiceResponse :definition-)
      "#request
geometry_msgs/PoseArray real_pos

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
---
#response
geometry_msgs/PoseArray sim_pos

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
")



(provide :uni_lace_msgs/UnityStateControllerService "f41682cf3f385f5d1be3d8588334a602")


