# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "sl_msgs: 0 messages, 3 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(sl_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" NAME_WE)
add_custom_target(_sl_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sl_msgs" "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" "geometry_msgs/PoseArray:geometry_msgs/Quaternion:std_msgs/Header:std_msgs/MultiArrayDimension:geometry_msgs/Pose:std_msgs/Float64MultiArray:std_msgs/MultiArrayLayout:geometry_msgs/Point"
)

get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" NAME_WE)
add_custom_target(_sl_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sl_msgs" "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" "std_msgs/Float32MultiArray:std_msgs/Float32:std_msgs/MultiArrayDimension:std_msgs/MultiArrayLayout:std_msgs/Int8:std_msgs/Int8MultiArray"
)

get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" NAME_WE)
add_custom_target(_sl_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sl_msgs" "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" "std_msgs/MultiArrayDimension:std_msgs/MultiArrayLayout:std_msgs/Int8:std_msgs/String:std_msgs/Int8MultiArray"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sl_msgs
)
_generate_srv_cpp(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sl_msgs
)
_generate_srv_cpp(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sl_msgs
)

### Generating Module File
_generate_module_cpp(sl_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sl_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(sl_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(sl_msgs_generate_messages sl_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_cpp _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_cpp _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_cpp _sl_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sl_msgs_gencpp)
add_dependencies(sl_msgs_gencpp sl_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sl_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sl_msgs
)
_generate_srv_eus(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sl_msgs
)
_generate_srv_eus(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sl_msgs
)

### Generating Module File
_generate_module_eus(sl_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sl_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(sl_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(sl_msgs_generate_messages sl_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_eus _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_eus _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_eus _sl_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sl_msgs_geneus)
add_dependencies(sl_msgs_geneus sl_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sl_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sl_msgs
)
_generate_srv_lisp(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sl_msgs
)
_generate_srv_lisp(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sl_msgs
)

### Generating Module File
_generate_module_lisp(sl_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sl_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(sl_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(sl_msgs_generate_messages sl_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_lisp _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_lisp _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_lisp _sl_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sl_msgs_genlisp)
add_dependencies(sl_msgs_genlisp sl_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sl_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sl_msgs
)
_generate_srv_nodejs(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sl_msgs
)
_generate_srv_nodejs(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sl_msgs
)

### Generating Module File
_generate_module_nodejs(sl_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sl_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(sl_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(sl_msgs_generate_messages sl_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_nodejs _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_nodejs _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_nodejs _sl_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sl_msgs_gennodejs)
add_dependencies(sl_msgs_gennodejs sl_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sl_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseArray.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs
)
_generate_srv_py(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs
)
_generate_srv_py(sl_msgs
  "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Int8MultiArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs
)

### Generating Module File
_generate_module_py(sl_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(sl_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(sl_msgs_generate_messages sl_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findTargetsService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_py _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPatternService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_py _sl_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/catkin_ws/src/robot_sl/sl_msgs/srv/findPlanService.srv" NAME_WE)
add_dependencies(sl_msgs_generate_messages_py _sl_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sl_msgs_genpy)
add_dependencies(sl_msgs_genpy sl_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sl_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sl_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sl_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(sl_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(sl_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sl_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sl_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(sl_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(sl_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sl_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sl_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(sl_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(sl_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sl_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sl_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(sl_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(sl_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sl_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(sl_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(sl_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
