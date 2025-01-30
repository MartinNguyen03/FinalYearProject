; Auto-generated. Do not edit!


(cl:in-package uni_lace_msgs-srv)


;//! \htmlinclude UniLaceParamService-request.msg.html

(cl:defclass <UniLaceParamService-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass UniLaceParamService-request (<UniLaceParamService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceParamService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceParamService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceParamService-request> is deprecated: use uni_lace_msgs-srv:UniLaceParamService-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceParamService-request>) ostream)
  "Serializes a message object of type '<UniLaceParamService-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceParamService-request>) istream)
  "Deserializes a message object of type '<UniLaceParamService-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceParamService-request>)))
  "Returns string type for a service object of type '<UniLaceParamService-request>"
  "uni_lace_msgs/UniLaceParamServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceParamService-request)))
  "Returns string type for a service object of type 'UniLaceParamService-request"
  "uni_lace_msgs/UniLaceParamServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceParamService-request>)))
  "Returns md5sum for a message object of type '<UniLaceParamService-request>"
  "dd5debcaac10dc60355cf969fc28100a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceParamService-request)))
  "Returns md5sum for a message object of type 'UniLaceParamService-request"
  "dd5debcaac10dc60355cf969fc28100a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceParamService-request>)))
  "Returns full string definition for message of type '<UniLaceParamService-request>"
  (cl:format cl:nil "#request~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceParamService-request)))
  "Returns full string definition for message of type 'UniLaceParamService-request"
  (cl:format cl:nil "#request~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceParamService-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceParamService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceParamService-request
))
;//! \htmlinclude UniLaceParamService-response.msg.html

(cl:defclass <UniLaceParamService-response> (roslisp-msg-protocol:ros-message)
  ((params_json
    :reader params_json
    :initarg :params_json
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass UniLaceParamService-response (<UniLaceParamService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UniLaceParamService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UniLaceParamService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name uni_lace_msgs-srv:<UniLaceParamService-response> is deprecated: use uni_lace_msgs-srv:UniLaceParamService-response instead.")))

(cl:ensure-generic-function 'params_json-val :lambda-list '(m))
(cl:defmethod params_json-val ((m <UniLaceParamService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader uni_lace_msgs-srv:params_json-val is deprecated.  Use uni_lace_msgs-srv:params_json instead.")
  (params_json m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UniLaceParamService-response>) ostream)
  "Serializes a message object of type '<UniLaceParamService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'params_json) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UniLaceParamService-response>) istream)
  "Deserializes a message object of type '<UniLaceParamService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'params_json) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UniLaceParamService-response>)))
  "Returns string type for a service object of type '<UniLaceParamService-response>"
  "uni_lace_msgs/UniLaceParamServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceParamService-response)))
  "Returns string type for a service object of type 'UniLaceParamService-response"
  "uni_lace_msgs/UniLaceParamServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UniLaceParamService-response>)))
  "Returns md5sum for a message object of type '<UniLaceParamService-response>"
  "dd5debcaac10dc60355cf969fc28100a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UniLaceParamService-response)))
  "Returns md5sum for a message object of type 'UniLaceParamService-response"
  "dd5debcaac10dc60355cf969fc28100a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UniLaceParamService-response>)))
  "Returns full string definition for message of type '<UniLaceParamService-response>"
  (cl:format cl:nil "#response~%std_msgs/String params_json~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UniLaceParamService-response)))
  "Returns full string definition for message of type 'UniLaceParamService-response"
  (cl:format cl:nil "#response~%std_msgs/String params_json~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UniLaceParamService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'params_json))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UniLaceParamService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UniLaceParamService-response
    (cl:cons ':params_json (params_json msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UniLaceParamService)))
  'UniLaceParamService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UniLaceParamService)))
  'UniLaceParamService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UniLaceParamService)))
  "Returns string type for a service object of type '<UniLaceParamService>"
  "uni_lace_msgs/UniLaceParamService")