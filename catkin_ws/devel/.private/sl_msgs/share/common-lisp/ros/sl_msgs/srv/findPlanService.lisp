; Auto-generated. Do not edit!


(cl:in-package sl_msgs-srv)


;//! \htmlinclude findPlanService-request.msg.html

(cl:defclass <findPlanService-request> (roslisp-msg-protocol:ros-message)
  ((pattern
    :reader pattern
    :initarg :pattern
    :type std_msgs-msg:Int8MultiArray
    :initform (cl:make-instance 'std_msgs-msg:Int8MultiArray))
   (aesthetic_mode
    :reader aesthetic_mode
    :initarg :aesthetic_mode
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass findPlanService-request (<findPlanService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <findPlanService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'findPlanService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sl_msgs-srv:<findPlanService-request> is deprecated: use sl_msgs-srv:findPlanService-request instead.")))

(cl:ensure-generic-function 'pattern-val :lambda-list '(m))
(cl:defmethod pattern-val ((m <findPlanService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:pattern-val is deprecated.  Use sl_msgs-srv:pattern instead.")
  (pattern m))

(cl:ensure-generic-function 'aesthetic_mode-val :lambda-list '(m))
(cl:defmethod aesthetic_mode-val ((m <findPlanService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:aesthetic_mode-val is deprecated.  Use sl_msgs-srv:aesthetic_mode instead.")
  (aesthetic_mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <findPlanService-request>) ostream)
  "Serializes a message object of type '<findPlanService-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pattern) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'aesthetic_mode) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <findPlanService-request>) istream)
  "Deserializes a message object of type '<findPlanService-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pattern) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'aesthetic_mode) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<findPlanService-request>)))
  "Returns string type for a service object of type '<findPlanService-request>"
  "sl_msgs/findPlanServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findPlanService-request)))
  "Returns string type for a service object of type 'findPlanService-request"
  "sl_msgs/findPlanServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<findPlanService-request>)))
  "Returns md5sum for a message object of type '<findPlanService-request>"
  "e752f9c85170cb73cc8cf790042cfd2d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'findPlanService-request)))
  "Returns md5sum for a message object of type 'findPlanService-request"
  "e752f9c85170cb73cc8cf790042cfd2d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<findPlanService-request>)))
  "Returns full string definition for message of type '<findPlanService-request>"
  (cl:format cl:nil "std_msgs/Int8MultiArray pattern~%std_msgs/Int8 aesthetic_mode~%~%================================================================================~%MSG: std_msgs/Int8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%int8[]            data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'findPlanService-request)))
  "Returns full string definition for message of type 'findPlanService-request"
  (cl:format cl:nil "std_msgs/Int8MultiArray pattern~%std_msgs/Int8 aesthetic_mode~%~%================================================================================~%MSG: std_msgs/Int8MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%int8[]            data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <findPlanService-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pattern))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'aesthetic_mode))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <findPlanService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'findPlanService-request
    (cl:cons ':pattern (pattern msg))
    (cl:cons ':aesthetic_mode (aesthetic_mode msg))
))
;//! \htmlinclude findPlanService-response.msg.html

(cl:defclass <findPlanService-response> (roslisp-msg-protocol:ros-message)
  ((plan
    :reader plan
    :initarg :plan
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass findPlanService-response (<findPlanService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <findPlanService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'findPlanService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sl_msgs-srv:<findPlanService-response> is deprecated: use sl_msgs-srv:findPlanService-response instead.")))

(cl:ensure-generic-function 'plan-val :lambda-list '(m))
(cl:defmethod plan-val ((m <findPlanService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sl_msgs-srv:plan-val is deprecated.  Use sl_msgs-srv:plan instead.")
  (plan m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <findPlanService-response>) ostream)
  "Serializes a message object of type '<findPlanService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'plan) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <findPlanService-response>) istream)
  "Deserializes a message object of type '<findPlanService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'plan) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<findPlanService-response>)))
  "Returns string type for a service object of type '<findPlanService-response>"
  "sl_msgs/findPlanServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findPlanService-response)))
  "Returns string type for a service object of type 'findPlanService-response"
  "sl_msgs/findPlanServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<findPlanService-response>)))
  "Returns md5sum for a message object of type '<findPlanService-response>"
  "e752f9c85170cb73cc8cf790042cfd2d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'findPlanService-response)))
  "Returns md5sum for a message object of type 'findPlanService-response"
  "e752f9c85170cb73cc8cf790042cfd2d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<findPlanService-response>)))
  "Returns full string definition for message of type '<findPlanService-response>"
  (cl:format cl:nil "std_msgs/String plan~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'findPlanService-response)))
  "Returns full string definition for message of type 'findPlanService-response"
  (cl:format cl:nil "std_msgs/String plan~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <findPlanService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'plan))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <findPlanService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'findPlanService-response
    (cl:cons ':plan (plan msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'findPlanService)))
  'findPlanService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'findPlanService)))
  'findPlanService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'findPlanService)))
  "Returns string type for a service object of type '<findPlanService>"
  "sl_msgs/findPlanService")