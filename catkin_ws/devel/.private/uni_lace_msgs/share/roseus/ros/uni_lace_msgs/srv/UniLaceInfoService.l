;; Auto-generated. Do not edit!


(when (boundp 'uni_lace_msgs::UniLaceInfoService)
  (if (not (find-package "UNI_LACE_MSGS"))
    (make-package "UNI_LACE_MSGS"))
  (shadow 'UniLaceInfoService (find-package "UNI_LACE_MSGS")))
(unless (find-package "UNI_LACE_MSGS::UNILACEINFOSERVICE")
  (make-package "UNI_LACE_MSGS::UNILACEINFOSERVICE"))
(unless (find-package "UNI_LACE_MSGS::UNILACEINFOSERVICEREQUEST")
  (make-package "UNI_LACE_MSGS::UNILACEINFOSERVICEREQUEST"))
(unless (find-package "UNI_LACE_MSGS::UNILACEINFOSERVICERESPONSE")
  (make-package "UNI_LACE_MSGS::UNILACEINFOSERVICERESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass uni_lace_msgs::UniLaceInfoServiceRequest
  :super ros::object
  :slots (_render ))

(defmethod uni_lace_msgs::UniLaceInfoServiceRequest
  (:init
   (&key
    ((:render __render) (instance std_msgs::Bool :init))
    )
   (send-super :init)
   (setq _render __render)
   self)
  (:render
   (&rest __render)
   (if (keywordp (car __render))
       (send* _render __render)
     (progn
       (if __render (setq _render (car __render)))
       _render)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Bool _render
    (send _render :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Bool _render
       (send _render :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Bool _render
     (send _render :deserialize buf ptr-) (incf ptr- (send _render :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UniLaceInfoServiceResponse
  :super ros::object
  :slots (_info ))

(defmethod uni_lace_msgs::UniLaceInfoServiceResponse
  (:init
   (&key
    ((:info __info) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _info __info)
   self)
  (:info
   (&rest __info)
   (if (keywordp (car __info))
       (send* _info __info)
     (progn
       (if __info (setq _info (car __info)))
       _info)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _info
    (send _info :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _info
       (send _info :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _info
     (send _info :deserialize buf ptr-) (incf ptr- (send _info :serialization-length))
   ;;
   self)
  )

(defclass uni_lace_msgs::UniLaceInfoService
  :super ros::object
  :slots ())

(setf (get uni_lace_msgs::UniLaceInfoService :md5sum-) "3a92e7221553a608fa193b6ae18fdde5")
(setf (get uni_lace_msgs::UniLaceInfoService :datatype-) "uni_lace_msgs/UniLaceInfoService")
(setf (get uni_lace_msgs::UniLaceInfoService :request) uni_lace_msgs::UniLaceInfoServiceRequest)
(setf (get uni_lace_msgs::UniLaceInfoService :response) uni_lace_msgs::UniLaceInfoServiceResponse)

(defmethod uni_lace_msgs::UniLaceInfoServiceRequest
  (:response () (instance uni_lace_msgs::UniLaceInfoServiceResponse :init)))

(setf (get uni_lace_msgs::UniLaceInfoServiceRequest :md5sum-) "3a92e7221553a608fa193b6ae18fdde5")
(setf (get uni_lace_msgs::UniLaceInfoServiceRequest :datatype-) "uni_lace_msgs/UniLaceInfoServiceRequest")
(setf (get uni_lace_msgs::UniLaceInfoServiceRequest :definition-)
      "#request
std_msgs/Bool render

================================================================================
MSG: std_msgs/Bool
---
#response
std_msgs/String info

================================================================================
MSG: std_msgs/String
string data
")

(setf (get uni_lace_msgs::UniLaceInfoServiceResponse :md5sum-) "3a92e7221553a608fa193b6ae18fdde5")
(setf (get uni_lace_msgs::UniLaceInfoServiceResponse :datatype-) "uni_lace_msgs/UniLaceInfoServiceResponse")
(setf (get uni_lace_msgs::UniLaceInfoServiceResponse :definition-)
      "#request
std_msgs/Bool render

================================================================================
MSG: std_msgs/Bool
---
#response
std_msgs/String info

================================================================================
MSG: std_msgs/String
string data
")



(provide :uni_lace_msgs/UniLaceInfoService "3a92e7221553a608fa193b6ae18fdde5")


