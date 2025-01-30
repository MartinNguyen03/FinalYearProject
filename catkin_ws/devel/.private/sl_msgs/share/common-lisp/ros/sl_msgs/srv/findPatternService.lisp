; Auto-generated. Do not edit!


(cl:in-package sl_msgs-srv)


;//! \htmlinclude findPatternService-request.msg.html

(cl:defclass <findPatternService-request> (roslisp-msg-protocol:ros-message)
  ((num_eyelets
    :reader num_eyelets
    :initarg :num_eyelets
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8))
   (shoelace_length
    :reader shoelace_length
    :initarg :shoelace_length
    :type std_msgs-msg:Float32
    :initform (cl:make-instance 'std_msgs-msg:Float32))
   (eyelet_positions
    :reader eyelet_positions
    :initarg :eyelet_positions
    :type std_msgs-msg:Float32MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Float32MultiArray)))
)

(cl:defclass findPatternService-request (<findPatternService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <findPatternService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'findPatternService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sl_msgs-srv:<findPatternService-request> is deprecated: use sl_msgs-srv:findPatternService-request instead.")))

(cl:ensure-generic-function 'num_eyelets-val :lambda-list '(m))
(cl:defmethod num_eyelets-val ((m <findPatternService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:num_eyelets-val is deprecated.  Use sl_msgs-srv:num_eyelets instead.")
  (num_eyelets m))

(cl:ensure-generic-function 'shoelace_length-val :lambda-list '(m))
(cl:defmethod shoelace_length-val ((m <findPatternService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:shoelace_length-val is deprecated.  Use sl_msgs-srv:shoelace_length instead.")
  (shoelace_length m))

(cl:ensure-generic-function 'eyelet_positions-val :lambda-list '(m))
(cl:defmethod eyelet_positions-val ((m <findPatternService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:eyelet_positions-val is deprecated.  Use sl_msgs-srv:eyelet_positions instead.")
  (eyelet_positions m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <findPatternService-request>) ostream)
  "Serializes a message object of type '<findPatternService-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'num_eyelets) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'shoelace_length) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'eyelet_positions) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <findPatternService-request>) istream)
  "Deserializes a message object of type '<findPatternService-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'num_eyelets) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'shoelace_length) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'eyelet_positions) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<findPatternService-request>)))
  "Returns string type for a service object of type '<findPatternService-request>"
  "sl_msgs/findPatternServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findPatternService-request)))
  "Returns string type for a service object of type 'findPatternService-request"
  "sl_msgs/findPatternServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<findPatternService-request>)))
  "Returns md5sum for a message object of type '<findPatternService-request>"
  "145ffa40f05356208146cb6e34948f0c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'findPatternService-request)))
  "Returns md5sum for a message object of type 'findPatternService-request"
  "145ffa40f05356208146cb6e34948f0c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<findPatternService-request>)))
  "Returns full string definition for message of type '<findPatternService-request>"
  (cl:format cl:nil "std_msgs/Int8 num_eyelets~%std_msgs/Float32 shoelace_length~%std_msgs/Float32MultiArray eyelet_positions~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'findPatternService-request)))
  "Returns full string definition for message of type 'findPatternService-request"
  (cl:format cl:nil "std_msgs/Int8 num_eyelets~%std_msgs/Float32 shoelace_length~%std_msgs/Float32MultiArray eyelet_positions~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <findPatternService-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'num_eyelets))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'shoelace_length))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'eyelet_positions))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <findPatternService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'findPatternService-request
    (cl:cons ':num_eyelets (num_eyelets msg))
    (cl:cons ':shoelace_length (shoelace_length msg))
    (cl:cons ':eyelet_positions (eyelet_positions msg))
))
;//! \htmlinclude findPatternService-response.msg.html

(cl:defclass <findPatternService-response> (roslisp-msg-protocol:ros-message)
  ((pattern
    :reader pattern
    :initarg :pattern
    :type std_msgs-msg:Int8MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Int8MultiArray)))
)

(cl:defclass findPatternService-response (<findPatternService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <findPatternService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'findPatternService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sl_msgs-srv:<findPatternService-response> is deprecated: use sl_msgs-srv:findPatternService-response instead.")))

(cl:ensure-generic-function 'pattern-val :lambda-list '(m))
(cl:defmethod pattern-val ((m <findPatternService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:pattern-val is deprecated.  Use sl_msgs-srv:pattern instead.")
  (pattern m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <findPatternService-response>) ostream)
  "Serializes a message object of type '<findPatternService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pattern) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <findPatternService-response>) istream)
  "Deserializes a message object of type '<findPatternService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pattern) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<findPatternService-response>)))
  "Returns string type for a service object of type '<findPatternService-response>"
  "sl_msgs/findPatternServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findPatternService-response)))
  "Returns string type for a service object of type 'findPatternService-response"
  "sl_msgs/findPatternServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<findPatternService-response>)))
  "Returns md5sum for a message object of type '<findPatternService-response>"
  "145ffa40f05356208146cb6e34948f0c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'findPatternService-response)))
  "Returns md5sum for a message object of type 'findPatternService-response"
  "145ffa40f05356208146cb6e34948f0c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<findPatternService-response>)))
  "Returns full string definition for message of type '<findPatternService-response>"
  (cl:format cl:nil "std_msgs/Int8MultiArray pattern~%~%================================================================================~%MSG: std_msgs/Int8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%int8[]            data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'findPatternService-response)))
  "Returns full string definition for message of type 'findPatternService-response"
  (cl:format cl:nil "std_msgs/Int8MultiArray pattern~%~%================================================================================~%MSG: std_msgs/Int8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%int8[]            data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <findPatternService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pattern))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <findPatternService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'findPatternService-response
    (cl:cons ':pattern (pattern msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'findPatternService)))
  'findPatternService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'findPatternService)))
  'findPatternService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findPatternService)))
  "Returns string type for a service object of type '<findPatternService>"
  "sl_msgs/findPatternService")