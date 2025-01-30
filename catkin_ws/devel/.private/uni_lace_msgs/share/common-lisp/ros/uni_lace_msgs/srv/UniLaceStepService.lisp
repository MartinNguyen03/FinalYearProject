; Auto-generated. Do not edit!


(cl:in-package uni_lace_msgs-srv)


;//! \htmlinclude UniLaceStepService-request.msg.html

(cl:defclass <UniLaceStepService-request> (roslisp-msg-protocol:ros-message)
  ((act
    :reader act
    :initarg :act
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass UniLaceStepService-request (<UniLaceStepService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceStepService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceStepService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceStepService-request> is deprecated: use uni_lace_msgs-srv:UniLaceStepService-request instead.")))

(cl:ensure-generic-function 'act-val :lambda-list '(m))
(cl:defmethod act-val ((m <UniLaceStepService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:act-val is deprecated.  Use uni_lace_msgs-srv:act instead.")
  (act m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceStepService-request>) ostream)
  "Serializes a message object of type '<UniLaceStepService-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'act) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceStepService-request>) istream)
  "Deserializes a message object of type '<UniLaceStepService-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'act) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceStepService-request>)))
  "Returns string type for a service object of type '<UniLaceStepService-request>"
  "uni_lace_msgs/UniLaceStepServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceStepService-request)))
  "Returns string type for a service object of type 'UniLaceStepService-request"
  "uni_lace_msgs/UniLaceStepServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceStepService-request>)))
  "Returns md5sum for a message object of type '<UniLaceStepService-request>"
  "af9f61a907e31117188b236c02aaf4bd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceStepService-request)))
  "Returns md5sum for a message object of type 'UniLaceStepService-request"
  "af9f61a907e31117188b236c02aaf4bd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceStepService-request>)))
  "Returns full string definition for message of type '<UniLaceStepService-request>"
  (cl:format cl:nil "#request~%std_msgs/String act~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceStepService-request)))
  "Returns full string definition for message of type 'UniLaceStepService-request"
  (cl:format cl:nil "#request~%std_msgs/String act~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceStepService-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'act))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceStepService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceStepService-request
    (cl:cons ':act (act msg))
))
;//! \htmlinclude UniLaceStepService-response.msg.html

(cl:defclass <UniLaceStepService-response> (roslisp-msg-protocol:ros-message)
  ((obs
    :reader obs
    :initarg :obs
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String))
   (obs_raw
    :reader obs_raw
    :initarg :obs_raw
    :type std_msgs-msg:UInt8MultiArray
    :initform (cl:make-instance 'std_msgs-msg:UInt8MultiArray)))
)

(cl:defclass UniLaceStepService-response (<UniLaceStepService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceStepService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceStepService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceStepService-response> is deprecated: use uni_lace_msgs-srv:UniLaceStepService-response instead.")))

(cl:ensure-generic-function 'obs-val :lambda-list '(m))
(cl:defmethod obs-val ((m <UniLaceStepService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:obs-val is deprecated.  Use uni_lace_msgs-srv:obs instead.")
  (obs m))

(cl:ensure-generic-function 'obs_raw-val :lambda-list '(m))
(cl:defmethod obs_raw-val ((m <UniLaceStepService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:obs_raw-val is deprecated.  Use uni_lace_msgs-srv:obs_raw instead.")
  (obs_raw m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceStepService-response>) ostream)
  "Serializes a message object of type '<UniLaceStepService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'obs) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'obs_raw) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceStepService-response>) istream)
  "Deserializes a message object of type '<UniLaceStepService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'obs) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'obs_raw) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceStepService-response>)))
  "Returns string type for a service object of type '<UniLaceStepService-response>"
  "uni_lace_msgs/UniLaceStepServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceStepService-response)))
  "Returns string type for a service object of type 'UniLaceStepService-response"
  "uni_lace_msgs/UniLaceStepServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceStepService-response>)))
  "Returns md5sum for a message object of type '<UniLaceStepService-response>"
  "af9f61a907e31117188b236c02aaf4bd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceStepService-response)))
  "Returns md5sum for a message object of type 'UniLaceStepService-response"
  "af9f61a907e31117188b236c02aaf4bd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceStepService-response>)))
  "Returns full string definition for message of type '<UniLaceStepService-response>"
  (cl:format cl:nil "#response~%std_msgs/String obs~%std_msgs/UInt8MultiArray obs_raw~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/UInt8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%uint8[]           data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceStepService-response)))
  "Returns full string definition for message of type 'UniLaceStepService-response"
  (cl:format cl:nil "#response~%std_msgs/String obs~%std_msgs/UInt8MultiArray obs_raw~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/UInt8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%uint8[]           data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceStepService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'obs))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'obs_raw))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceStepService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceStepService-response
    (cl:cons ':obs (obs msg))
    (cl:cons ':obs_raw (obs_raw msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UniLaceStepService)))
  'UniLaceStepService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UniLaceStepService)))
  'UniLaceStepService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceStepService)))
  "Returns string type for a service object of type '<UniLaceStepService>"
  "uni_lace_msgs/UniLaceStepService")