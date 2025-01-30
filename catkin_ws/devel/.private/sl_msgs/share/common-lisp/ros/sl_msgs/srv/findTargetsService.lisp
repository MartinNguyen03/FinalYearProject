; Auto-generated. Do not edit!


(cl:in-package sl_msgs-srv)


;//! \htmlinclude findTargetsService-request.msg.html

(cl:defclass <findTargetsService-request> (roslisp-msg-protocol:ros-message)
  ((target_name
    :reader target_name
    :initarg :target_name
    :type cl:string
    :initform "")
   (camera_name
    :reader camera_name
    :initarg :camera_name
    :type cl:string
    :initform ""))
)

(cl:defclass findTargetsService-request (<findTargetsService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <findTargetsService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'findTargetsService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sl_msgs-srv:<findTargetsService-request> is deprecated: use sl_msgs-srv:findTargetsService-request instead.")))

(cl:ensure-generic-function 'target_name-val :lambda-list '(m))
(cl:defmethod target_name-val ((m <findTargetsService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:target_name-val is deprecated.  Use sl_msgs-srv:target_name instead.")
  (target_name m))

(cl:ensure-generic-function 'camera_name-val :lambda-list '(m))
(cl:defmethod camera_name-val ((m <findTargetsService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:camera_name-val is deprecated.  Use sl_msgs-srv:camera_name instead.")
  (camera_name m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <findTargetsService-request>) ostream)
  "Serializes a message object of type '<findTargetsService-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'target_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'target_name))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'camera_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'camera_name))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <findTargetsService-request>) istream)
  "Deserializes a message object of type '<findTargetsService-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'target_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'target_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'camera_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'camera_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<findTargetsService-request>)))
  "Returns string type for a service object of type '<findTargetsService-request>"
  "sl_msgs/findTargetsServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findTargetsService-request)))
  "Returns string type for a service object of type 'findTargetsService-request"
  "sl_msgs/findTargetsServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<findTargetsService-request>)))
  "Returns md5sum for a message object of type '<findTargetsService-request>"
  "aa823e51a177fa5a87a4dc28394b4618")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'findTargetsService-request)))
  "Returns md5sum for a message object of type 'findTargetsService-request"
  "aa823e51a177fa5a87a4dc28394b4618")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<findTargetsService-request>)))
  "Returns full string definition for message of type '<findTargetsService-request>"
  (cl:format cl:nil "string target_name~%string camera_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'findTargetsService-request)))
  "Returns full string definition for message of type 'findTargetsService-request"
  (cl:format cl:nil "string target_name~%string camera_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <findTargetsService-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'target_name))
     4 (cl:length (cl:slot-value msg 'camera_name))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <findTargetsService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'findTargetsService-request
    (cl:cons ':target_name (target_name msg))
    (cl:cons ':camera_name (camera_name msg))
))
;//! \htmlinclude findTargetsService-response.msg.html

(cl:defclass <findTargetsService-response> (roslisp-msg-protocol:ros-message)
  ((target
    :reader target
    :initarg :target
    :type geometry_msgs-msg:PoseArray
    :initform (cl:make-instance 'geometry_msgs-msg:PoseArray))
   (confidence
    :reader confidence
    :initarg :confidence
    :type std_msgs-msg:Float64MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Float64MultiArray)))
)

(cl:defclass findTargetsService-response (<findTargetsService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <findTargetsService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'findTargetsService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sl_msgs-srv:<findTargetsService-response> is deprecated: use sl_msgs-srv:findTargetsService-response instead.")))

(cl:ensure-generic-function 'target-val :lambda-list '(m))
(cl:defmethod target-val ((m <findTargetsService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:target-val is deprecated.  Use sl_msgs-srv:target instead.")
  (target m))

(cl:ensure-generic-function 'confidence-val :lambda-list '(m))
(cl:defmethod confidence-val ((m <findTargetsService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:confidence-val is deprecated.  Use sl_msgs-srv:confidence instead.")
  (confidence m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <findTargetsService-response>) ostream)
  "Serializes a message object of type '<findTargetsService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'target) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'confidence) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <findTargetsService-response>) istream)
  "Deserializes a message object of type '<findTargetsService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'target) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'confidence) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<findTargetsService-response>)))
  "Returns string type for a service object of type '<findTargetsService-response>"
  "sl_msgs/findTargetsServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findTargetsService-response)))
  "Returns string type for a service object of type 'findTargetsService-response"
  "sl_msgs/findTargetsServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<findTargetsService-response>)))
  "Returns md5sum for a message object of type '<findTargetsService-response>"
  "aa823e51a177fa5a87a4dc28394b4618")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'findTargetsService-response)))
  "Returns md5sum for a message object of type 'findTargetsService-response"
  "aa823e51a177fa5a87a4dc28394b4618")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<findTargetsService-response>)))
  "Returns full string definition for message of type '<findTargetsService-response>"
  (cl:format cl:nil "geometry_msgs/PoseArray target~%std_msgs/Float64MultiArray confidence~%~%================================================================================~%MSG: geometry_msgs/PoseArray~%# An array of poses with a header for global reference.~%~%Header header~%~%Pose[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: std_msgs/Float64MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float64[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'findTargetsService-response)))
  "Returns full string definition for message of type 'findTargetsService-response"
  (cl:format cl:nil "geometry_msgs/PoseArray target~%std_msgs/Float64MultiArray confidence~%~%================================================================================~%MSG: geometry_msgs/PoseArray~%# An array of poses with a header for global reference.~%~%Header header~%~%Pose[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: std_msgs/Float64MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float64[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <findTargetsService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'target))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'confidence))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <findTargetsService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'findTargetsService-response
    (cl:cons ':target (target msg))
    (cl:cons ':confidence (confidence msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'findTargetsService)))
  'findTargetsService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'findTargetsService)))
  'findTargetsService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findTargetsService)))
  "Returns string type for a service object of type '<findTargetsService>"
  "sl_msgs/findTargetsService")