;; Auto-generated. Do not edit!


(when (boundp 'uni_lace_msgs::UniLaceParamService)
  (if (not (find-package "UNI_LACE_MSGS"))
    (make-package "UNI_LACE_MSGS"))
  (shadow 'UniLaceParamService (find-package "UNI_LACE_MSGS")))
(unless (find-package "UNI_LACE_MSGS::UNILACEPARAMSERVICE")
  (make-package "UNI_LACE_MSGS::UNILACEPARAMSERVICE"))
(unless (find-package "UNI_LACE_MSGS::UNILACEPARAMSERVICEREQUEST")
  (make-package "UNI_LACE_MSGS::UNILACEPARAMSERVICEREQUEST"))
(unless (find-package "UNI_LACE_MSGS::UNILACEPARAMSERVICERESPONSE")
  (make-package "UNI_LACE_MSGS::UNILACEPARAMSERVICERESPONSE"))

(in-package "ROS")



(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass uni_lace_msgs::UniLaceParamServiceRequest
  :super ros::object
  :slots ())

(defmethod uni_lace_msgs::UniLaceParamServiceRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass uni_lace_msgs::UniLaceParamServiceResponse
  :super ros::object
  :slots (_params_json ))

(defmethod uni_lace_msgs::UniLaceParamServiceResponse
  (:init
   (&key
    ((:params_json __params_json) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _params_json __params_json)
   self)
  (:params_json
   (&rest __params_json)
   (if (keywordp (car __params_json))
       (send* _params_json __params_json)
     (progn
       (if __params_json (setq _params_json (car __params_json)))
       _params_json)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _params_json
    (send _params_json :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _params_json
       (send _params_json :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _params_json
     (send _params_json :deserialize buf ptr-) (incf ptr- (send _params_json :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UniLaceParamService
  :super ros::object
  :slots ())

(setf (get uni_lace_msgs::UniLaceParamService :md5sum-) "dd5debcaac10dc60355cf969fc28100a")
(setf (get uni_lace_msgs::UniLaceParamService :datatype-) "uni_lace_msgs/UniLaceParamService")
(setf (get uni_lace_msgs::UniLaceParamService :request) uni_lace_msgs::UniLaceParamServiceRequest)
(setf (get uni_lace_msgs::UniLaceParamService :response) uni_lace_msgs::UniLaceParamServiceResponse)

(defmethod uni_lace_msgs::UniLaceParamServiceRequest
  (:response () (instance uni_lace_msgs::UniLaceParamServiceResponse :init)))

(setf (get uni_lace_msgs::UniLaceParamServiceRequest :md5sum-) "dd5debcaac10dc60355cf969fc28100a")
(setf (get uni_lace_msgs::UniLaceParamServiceRequest :datatype-) "uni_lace_msgs/UniLaceParamServiceRequest")
(setf (get uni_lace_msgs::UniLaceParamServiceRequest :definition-)
      "#request
---
#response
std_msgs/String params_json

================================================================================
MSG: std_msgs/String
string data
")

(setf (get uni_lace_msgs::UniLaceParamServiceResponse :md5sum-) "dd5debcaac10dc60355cf969fc28100a")
(setf (get uni_lace_msgs::UniLaceParamServiceResponse :datatype-) "uni_lace_msgs/UniLaceParamServiceResponse")
(setf (get uni_lace_msgs::UniLaceParamServiceResponse :definition-)
      "#request
---
#response
std_msgs/String params_json

================================================================================
MSG: std_msgs/String
string data
")



(provide :uni_lace_msgs/UniLaceParamService "dd5debcaac10dc60355cf969fc28100a")


