// Auto-generated. Do not edit!

// (in-package uni_lace_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class UnityStateControllerServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.real_pos = null;
    }
    else {
      if (initObj.hasOwnProperty('real_pos')) {
        this.real_pos = initObj.real_pos
      }
      else {
        this.real_pos = new geometry_msgs.msg.PoseArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UnityStateControllerServiceRequest
    // Serialize message field [real_pos]
    bufferOffset = geometry_msgs.msg.PoseArray.serialize(obj.real_pos, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UnityStateControllerServiceRequest
    let len;
    let data = new UnityStateControllerServiceRequest(null);
    // Deserialize message field [real_pos]
    data.real_pos = geometry_msgs.msg.PoseArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += geometry_msgs.msg.PoseArray.getMessageSize(object.real_pos);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UnityStateControllerServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3799f5ed1fbf96ab567652c8facf087a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #request
    geometry_msgs/PoseArray real_pos
    
    ================================================================================
    MSG: geometry_msgs/PoseArray
    # An array of poses with a header for global reference.
    
    Header header
    
    Pose[] poses
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new UnityStateControllerServiceRequest(null);
    if (msg.real_pos !== undefined) {
      resolved.real_pos = geometry_msgs.msg.PoseArray.Resolve(msg.real_pos)
    }
    else {
      resolved.real_pos = new geometry_msgs.msg.PoseArray()
    }

    return resolved;
    }
};

class UnityStateControllerServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.sim_pos = null;
    }
    else {
      if (initObj.hasOwnProperty('sim_pos')) {
        this.sim_pos = initObj.sim_pos
      }
      else {
        this.sim_pos = new geometry_msgs.msg.PoseArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UnityStateControllerServiceResponse
    // Serialize message field [sim_pos]
    bufferOffset = geometry_msgs.msg.PoseArray.serialize(obj.sim_pos, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UnityStateControllerServiceResponse
    let len;
    let data = new UnityStateControllerServiceResponse(null);
    // Deserialize message field [sim_pos]
    data.sim_pos = geometry_msgs.msg.PoseArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += geometry_msgs.msg.PoseArray.getMessageSize(object.sim_pos);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UnityStateControllerServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '15d431186ba988ebc556e3c1bad087a5';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #response
    geometry_msgs/PoseArray sim_pos
    
    ================================================================================
    MSG: geometry_msgs/PoseArray
    # An array of poses with a header for global reference.
    
    Header header
    
    Pose[] poses
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new UnityStateControllerServiceResponse(null);
    if (msg.sim_pos !== undefined) {
      resolved.sim_pos = geometry_msgs.msg.PoseArray.Resolve(msg.sim_pos)
    }
    else {
      resolved.sim_pos = new geometry_msgs.msg.PoseArray()
    }

    return resolved;
    }
};

module.exports = {
  Request: UnityStateControllerServiceRequest,
  Response: UnityStateControllerServiceResponse,
  md5sum() { return 'f41682cf3f385f5d1be3d8588334a602'; },
  datatype() { return 'uni_lace_msgs/UnityStateControllerService'; }
};
