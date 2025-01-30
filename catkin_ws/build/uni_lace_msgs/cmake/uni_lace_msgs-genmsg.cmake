# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "uni_lace_msgs: 0 messages, 5 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Itrajectory_msgs:/opt/ros/noetic/share/trajectory_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(uni_lace_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" NAME_WE)
add_custom_target(_uni_lace_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uni_lace_msgs" "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" "std_msgs/String"
)

get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" NAME_WE)
add_custom_target(_uni_lace_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uni_lace_msgs" "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" "std_msgs/MultiArrayLayout:std_msgs/MultiArrayDimension:std_msgs/String:std_msgs/UInt8MultiArray"
)

get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" NAME_WE)
add_custom_target(_uni_lace_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uni_lace_msgs" "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" "std_msgs/MultiArrayLayout:std_msgs/MultiArrayDimension:std_msgs/String:std_msgs/UInt8MultiArray"
)

get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" NAME_WE)
add_custom_target(_uni_lace_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uni_lace_msgs" "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" "std_msgs/String:std_msgs/Bool"
)

get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" NAME_WE)
add_custom_target(_uni_lace_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "uni_lace_msgs" "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" "geometry_msgs/Pose:std_msgs/Header:geometry_msgs/Point:geometry_msgs/PoseArray:geometry_msgs/Quaternion"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_cpp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_cpp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_cpp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Bool.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_cpp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
)

### Generating Module File
_generate_module_cpp(uni_lace_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(uni_lace_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(uni_lace_msgs_generate_messages uni_lace_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_cpp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_cpp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_cpp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_cpp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_cpp _uni_lace_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uni_lace_msgs_gencpp)
add_dependencies(uni_lace_msgs_gencpp uni_lace_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uni_lace_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_eus(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_eus(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_eus(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Bool.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_eus(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
)

### Generating Module File
_generate_module_eus(uni_lace_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(uni_lace_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(uni_lace_msgs_generate_messages uni_lace_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_eus _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_eus _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_eus _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_eus _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_eus _uni_lace_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uni_lace_msgs_geneus)
add_dependencies(uni_lace_msgs_geneus uni_lace_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uni_lace_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_lisp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_lisp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_lisp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Bool.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_lisp(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
)

### Generating Module File
_generate_module_lisp(uni_lace_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(uni_lace_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(uni_lace_msgs_generate_messages uni_lace_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_lisp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_lisp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_lisp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_lisp _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_lisp _uni_lace_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uni_lace_msgs_genlisp)
add_dependencies(uni_lace_msgs_genlisp uni_lace_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uni_lace_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_nodejs(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_nodejs(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_nodejs(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Bool.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_nodejs(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
)

### Generating Module File
_generate_module_nodejs(uni_lace_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(uni_lace_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(uni_lace_msgs_generate_messages uni_lace_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_nodejs _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_nodejs _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_nodejs _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_nodejs _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_nodejs _uni_lace_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uni_lace_msgs_gennodejs)
add_dependencies(uni_lace_msgs_gennodejs uni_lace_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uni_lace_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_py(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_py(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/UInt8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_py(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Bool.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
)
_generate_srv_py(uni_lace_msgs
  "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
)

### Generating Module File
_generate_module_py(uni_lace_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(uni_lace_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(uni_lace_msgs_generate_messages uni_lace_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceParamService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_py _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceStepService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_py _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceResetService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_py _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UniLaceInfoService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_py _uni_lace_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/uni_lace/uni_lace_msgs/srv/UnityStateControllerService.srv" NAME_WE)
add_dependencies(uni_lace_msgs_generate_messages_py _uni_lace_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(uni_lace_msgs_genpy)
add_dependencies(uni_lace_msgs_genpy uni_lace_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS uni_lace_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/uni_lace_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(uni_lace_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(uni_lace_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(uni_lace_msgs_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET trajectory_msgs_generate_messages_cpp)
  add_dependencies(uni_lace_msgs_generate_messages_cpp trajectory_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/uni_lace_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(uni_lace_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(uni_lace_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(uni_lace_msgs_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET trajectory_msgs_generate_messages_eus)
  add_dependencies(uni_lace_msgs_generate_messages_eus trajectory_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/uni_lace_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(uni_lace_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(uni_lace_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(uni_lace_msgs_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET trajectory_msgs_generate_messages_lisp)
  add_dependencies(uni_lace_msgs_generate_messages_lisp trajectory_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/uni_lace_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(uni_lace_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(uni_lace_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(uni_lace_msgs_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET trajectory_msgs_generate_messages_nodejs)
  add_dependencies(uni_lace_msgs_generate_messages_nodejs trajectory_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/uni_lace_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(uni_lace_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(uni_lace_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(uni_lace_msgs_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET trajectory_msgs_generate_messages_py)
  add_dependencies(uni_lace_msgs_generate_messages_py trajectory_msgs_generate_messages_py)
endif()
