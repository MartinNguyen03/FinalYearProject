; Auto-generated. Do not edit!


(cl:in-package uni_lace_msgs-srv)


;//! \htmlinclude UnityStateControllerService-request.msg.html

(cl:defclass <UnityStateControllerService-request> (roslisp-msg-protocol:ros-message)
  ((real_pos
    :reader real_pos
    :initarg :real_pos
    :type geometry_msgs-msg:PoseArray
    :initform (cl:make-instance 'geometry_msgs-msg:PoseArray)))
)

(cl:defclass UnityStateControllerService-request (<UnityStateControllerService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UnityStateControllerService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UnityStateControllerService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UnityStateControllerService-request> is deprecated: use uni_lace_msgs-srv:UnityStateControllerService-request instead.")))

(cl:ensure-generic-function 'real_pos-val :lambda-list '(m))
(cl:defmethod real_pos-val ((m <UnityStateControllerService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:real_pos-val is deprecated.  Use uni_lace_msgs-srv:real_pos instead.")
  (real_pos m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UnityStateControllerService-request>) ostream)
  "Serializes a message object of type '<UnityStateControllerService-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'real_pos) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UnityStateControllerService-request>) istream)
  "Deserializes a message object of type '<UnityStateControllerService-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'real_pos) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UnityStateControllerService-request>)))
  "Returns string type for a service object of type '<UnityStateControllerService-request>"
  "uni_lace_msgs/UnityStateControllerServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UnityStateControllerService-request)))
  "Returns string type for a service object of type 'UnityStateControllerService-request"
  "uni_lace_msgs/UnityStateControllerServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UnityStateControllerService-request>)))
  "Returns md5sum for a message object of type '<UnityStateControllerService-request>"
  "f41682cf3f385f5d1be3d8588334a602")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UnityStateControllerService-request)))
  "Returns md5sum for a message object of type 'UnityStateControllerService-request"
  "f41682cf3f385f5d1be3d8588334a602")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UnityStateControllerService-request>)))
  "Returns full string definition for message of type '<UnityStateControllerService-request>"
  (cl:format cl:nil "#request~%geometry_msgs/PoseArray real_pos~%~%================================================================================~%MSG: geometry_msgs/PoseArray~%# An array of poses with a header for global reference.~%~%Header header~%~%Pose[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UnityStateControllerService-request)))
  "Returns full string definition for message of type 'UnityStateControllerService-request"
  (cl:format cl:nil "#request~%geometry_msgs/PoseArray real_pos~%~%================================================================================~%MSG: geometry_msgs/PoseArray~%# An array of poses with a header for global reference.~%~%Header header~%~%Pose[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UnityStateControllerService-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'real_pos))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UnityStateControllerService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UnityStateControllerService-request
    (cl:cons ':real_pos (real_pos msg))
))
;//! \htmlinclude UnityStateControllerService-response.msg.html

(cl:defclass <UnityStateControllerService-response> (roslisp-msg-protocol:ros-message)
  ((sim_pos
    :reader sim_pos
    :initarg :sim_pos
    :type geometry_msgs-msg:PoseArray
    :initform (cl:make-instance 'geometry_msgs-msg:PoseArray)))
)

(cl:defclass UnityStateControllerService-response (<UnityStateControllerService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UnityStateControllerService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UnityStateControllerService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UnityStateControllerService-response> is deprecated: use uni_lace_msgs-srv:UnityStateControllerService-response instead.")))

(cl:ensure-generic-function 'sim_pos-val :lambda-list '(m))
(cl:defmethod sim_pos-val ((m <UnityStateControllerService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:sim_pos-val is deprecated.  Use uni_lace_msgs-srv:sim_pos instead.")
  (sim_pos m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UnityStateControllerService-response>) ostream)
  "Serializes a message object of type '<UnityStateControllerService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'sim_pos) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UnityStateControllerService-response>) istream)
  "Deserializes a message object of type '<UnityStateControllerService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'sim_pos) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UnityStateControllerService-response>)))
  "Returns string type for a service object of type '<UnityStateControllerService-response>"
  "uni_lace_msgs/UnityStateControllerServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UnityStateControllerService-response)))
  "Returns string type for a service object of type 'UnityStateControllerService-response"
  "uni_lace_msgs/UnityStateControllerServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UnityStateControllerService-response>)))
  "Returns md5sum for a message object of type '<UnityStateControllerService-response>"
  "f41682cf3f385f5d1be3d8588334a602")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UnityStateControllerService-response)))
  "Returns md5sum for a message object of type 'UnityStateControllerService-response"
  "f41682cf3f385f5d1be3d8588334a602")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UnityStateControllerService-response>)))
  "Returns full string definition for message of type '<UnityStateControllerService-response>"
  (cl:format cl:nil "#response~%geometry_msgs/PoseArray sim_pos~%~%================================================================================~%MSG: geometry_msgs/PoseArray~%# An array of poses with a header for global reference.~%~%Header header~%~%Pose[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UnityStateControllerService-response)))
  "Returns full string definition for message of type 'UnityStateControllerService-response"
  (cl:format cl:nil "#response~%geometry_msgs/PoseArray sim_pos~%~%================================================================================~%MSG: geometry_msgs/PoseArray~%# An array of poses with a header for global reference.~%~%Header header~%~%Pose[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UnityStateControllerService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'sim_pos))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UnityStateControllerService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UnityStateControllerService-response
    (cl:cons ':sim_pos (sim_pos msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UnityStateControllerService)))
  'UnityStateControllerService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UnityStateControllerService)))
  'UnityStateControllerService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UnityStateControllerService)))
  "Returns string type for a service object of type '<UnityStateControllerService>"
  "uni_lace_msgs/UnityStateControllerService")