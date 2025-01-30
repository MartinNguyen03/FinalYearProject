; Auto-generated. Do not edit!


(cl:in-package uni_lace_msgs-srv)


;//! \htmlinclude UniLaceResetService-request.msg.html

(cl:defclass <UniLaceResetService-request> (roslisp-msg-protocol:ros-message)
  ((params
    :reader params
    :initarg :params
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass UniLaceResetService-request (<UniLaceResetService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceResetService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceResetService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceResetService-request> is deprecated: use uni_lace_msgs-srv:UniLaceResetService-request instead.")))

(cl:ensure-generic-function 'params-val :lambda-list '(m))
(cl:defmethod params-val ((m <UniLaceResetService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:params-val is deprecated.  Use uni_lace_msgs-srv:params instead.")
  (params m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceResetService-request>) ostream)
  "Serializes a message object of type '<UniLaceResetService-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'params) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceResetService-request>) istream)
  "Deserializes a message object of type '<UniLaceResetService-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'params) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceResetService-request>)))
  "Returns string type for a service object of type '<UniLaceResetService-request>"
  "uni_lace_msgs/UniLaceResetServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceResetService-request)))
  "Returns string type for a service object of type 'UniLaceResetService-request"
  "uni_lace_msgs/UniLaceResetServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceResetService-request>)))
  "Returns md5sum for a message object of type '<UniLaceResetService-request>"
  "f4d24b83badf9180317ef840cefaa307")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceResetService-request)))
  "Returns md5sum for a message object of type 'UniLaceResetService-request"
  "f4d24b83badf9180317ef840cefaa307")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceResetService-request>)))
  "Returns full string definition for message of type '<UniLaceResetService-request>"
  (cl:format cl:nil "#request~%std_msgs/String params~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceResetService-request)))
  "Returns full string definition for message of type 'UniLaceResetService-request"
  (cl:format cl:nil "#request~%std_msgs/String params~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceResetService-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'params))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceResetService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceResetService-request
    (cl:cons ':params (params msg))
))
;//! \htmlinclude UniLaceResetService-response.msg.html

(cl:defclass <UniLaceResetService-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass UniLaceResetService-response (<UniLaceResetService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceResetService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceResetService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceResetService-response> is deprecated: use uni_lace_msgs-srv:UniLaceResetService-response instead.")))

(cl:ensure-generic-function 'obs-val :lambda-list '(m))
(cl:defmethod obs-val ((m <UniLaceResetService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:obs-val is deprecated.  Use uni_lace_msgs-srv:obs instead.")
  (obs m))

(cl:ensure-generic-function 'obs_raw-val :lambda-list '(m))
(cl:defmethod obs_raw-val ((m <UniLaceResetService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:obs_raw-val is deprecated.  Use uni_lace_msgs-srv:obs_raw instead.")
  (obs_raw m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceResetService-response>) ostream)
  "Serializes a message object of type '<UniLaceResetService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'obs) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'obs_raw) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceResetService-response>) istream)
  "Deserializes a message object of type '<UniLaceResetService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'obs) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'obs_raw) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceResetService-response>)))
  "Returns string type for a service object of type '<UniLaceResetService-response>"
  "uni_lace_msgs/UniLaceResetServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceResetService-response)))
  "Returns string type for a service object of type 'UniLaceResetService-response"
  "uni_lace_msgs/UniLaceResetServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceResetService-response>)))
  "Returns md5sum for a message object of type '<UniLaceResetService-response>"
  "f4d24b83badf9180317ef840cefaa307")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceResetService-response)))
  "Returns md5sum for a message object of type 'UniLaceResetService-response"
  "f4d24b83badf9180317ef840cefaa307")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceResetService-response>)))
  "Returns full string definition for message of type '<UniLaceResetService-response>"
  (cl:format cl:nil "#response~%std_msgs/String obs~%std_msgs/UInt8MultiArray obs_raw~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/UInt8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%uint8[]           data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceResetService-response)))
  "Returns full string definition for message of type 'UniLaceResetService-response"
  (cl:format cl:nil "#response~%std_msgs/String obs~%std_msgs/UInt8MultiArray obs_raw~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/UInt8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%uint8[]           data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceResetService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'obs))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'obs_raw))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceResetService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceResetService-response
    (cl:cons ':obs (obs msg))
    (cl:cons ':obs_raw (obs_raw msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UniLaceResetService)))
  'UniLaceResetService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UniLaceResetService)))
  'UniLaceResetService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceResetService)))
  "Returns string type for a service object of type '<UniLaceResetService>"
  "uni_lace_msgs/UniLaceResetService")