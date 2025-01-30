
(cl:in-package :asdf)

(defsystem "sl_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "findPatternService" :depends-on ("_package_findPatternService"))
    (:file "_package_findPatternService" :depends-on ("_package"))
    (:file "findPlanService" :depends-on ("_package_findPlanService"))
    (:file "_package_findPlanService" :depends-on ("_package"))
    (:file "findTargetsService" :depends-on ("_package_findTargetsService"))
    (:file "_package_findTargetsService" :depends-on ("_package"))
  ))