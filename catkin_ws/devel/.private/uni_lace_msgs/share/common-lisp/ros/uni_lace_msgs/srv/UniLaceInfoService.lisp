; Auto-generated. Do not edit!


(cl:in-package uni_lace_msgs-srv)


;//! \htmlinclude UniLaceInfoService-request.msg.html

(cl:defclass <UniLaceInfoService-request> (roslisp-msg-protocol:ros-message)
  ((render
    :reader render
    :initarg :render
    :type std_msgs-msg:Bool
    :initform (cl:make-instance 'std_msgs-msg:Bool)))
)

(cl:defclass UniLaceInfoService-request (<UniLaceInfoService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceInfoService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceInfoService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceInfoService-request> is deprecated: use uni_lace_msgs-srv:UniLaceInfoService-request instead.")))

(cl:ensure-generic-function 'render-val :lambda-list '(m))
(cl:defmethod render-val ((m <UniLaceInfoService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:render-val is deprecated.  Use uni_lace_msgs-srv:render instead.")
  (render m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceInfoService-request>) ostream)
  "Serializes a message object of type '<UniLaceInfoService-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'render) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceInfoService-request>) istream)
  "Deserializes a message object of type '<UniLaceInfoService-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'render) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceInfoService-request>)))
  "Returns string type for a service object of type '<UniLaceInfoService-request>"
  "uni_lace_msgs/UniLaceInfoServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceInfoService-request)))
  "Returns string type for a service object of type 'UniLaceInfoService-request"
  "uni_lace_msgs/UniLaceInfoServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceInfoService-request>)))
  "Returns md5sum for a message object of type '<UniLaceInfoService-request>"
  "3a92e7221553a608fa193b6ae18fdde5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceInfoService-request)))
  "Returns md5sum for a message object of type 'UniLaceInfoService-request"
  "3a92e7221553a608fa193b6ae18fdde5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceInfoService-request>)))
  "Returns full string definition for message of type '<UniLaceInfoService-request>"
  (cl:format cl:nil "#request~%std_msgs/Bool render~%~%================================================================================~%MSG: std_msgs/Bool~%bool data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceInfoService-request)))
  "Returns full string definition for message of type 'UniLaceInfoService-request"
  (cl:format cl:nil "#request~%std_msgs/Bool render~%~%================================================================================~%MSG: std_msgs/Bool~%bool data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceInfoService-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'render))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceInfoService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceInfoService-request
    (cl:cons ':render (render msg))
))
;//! \htmlinclude UniLaceInfoService-response.msg.html

(cl:defclass <UniLaceInfoService-response> (roslisp-msg-protocol:ros-message)
  ((info
    :reader info
    :initarg :info
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass UniLaceInfoService-response (<UniLaceInfoService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceInfoService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceInfoService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceInfoService-response> is deprecated: use uni_lace_msgs-srv:UniLaceInfoService-response instead.")))

(cl:ensure-generic-function 'info-val :lambda-list '(m))
(cl:defmethod info-val ((m <UniLaceInfoService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:info-val is deprecated.  Use uni_lace_msgs-srv:info instead.")
  (info m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceInfoService-response>) ostream)
  "Serializes a message object of type '<UniLaceInfoService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'info) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceInfoService-response>) istream)
  "Deserializes a message object of type '<UniLaceInfoService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'info) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceInfoService-response>)))
  "Returns string type for a service object of type '<UniLaceInfoService-response>"
  "uni_lace_msgs/UniLaceInfoServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceInfoService-response)))
  "Returns string type for a service object of type 'UniLaceInfoService-response"
  "uni_lace_msgs/UniLaceInfoServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceInfoService-response>)))
  "Returns md5sum for a message object of type '<UniLaceInfoService-response>"
  "3a92e7221553a608fa193b6ae18fdde5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceInfoService-response)))
  "Returns md5sum for a message object of type 'UniLaceInfoService-response"
  "3a92e7221553a608fa193b6ae18fdde5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceInfoService-response>)))
  "Returns full string definition for message of type '<UniLaceInfoService-response>"
  (cl:format cl:nil "#response~%std_msgs/String info~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceInfoService-response)))
  "Returns full string definition for message of type 'UniLaceInfoService-response"
  (cl:format cl:nil "#response~%std_msgs/String info~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceInfoService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'info))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceInfoService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceInfoService-response
    (cl:cons ':info (info msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UniLaceInfoService)))
  'UniLaceInfoService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UniLaceInfoService)))
  'UniLaceInfoService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceInfoService)))
  "Returns string type for a service object of type '<UniLaceInfoService>"
  "uni_lace_msgs/UniLaceInfoService")