// Auto-generated. Do not edit!

// (in-package sl_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class findPlanServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.pattern = null;
      this.aesthetic_mode = null;
    }
    else {
      if (initObj.hasOwnProperty('pattern')) {
        this.pattern = initObj.pattern
      }
      else {
        this.pattern = new std_msgs.msg.Int8MultiArray();
      }
      if (initObj.hasOwnProperty('aesthetic_mode')) {
        this.aesthetic_mode = initObj.aesthetic_mode
      }
      else {
        this.aesthetic_mode = new std_msgs.msg.Int8();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type findPlanServiceRequest
    // Serialize message field [pattern]
    bufferOffset = std_msgs.msg.Int8MultiArray.serialize(obj.pattern, buffer, bufferOffset);
    // Serialize message field [aesthetic_mode]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.aesthetic_mode, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type findPlanServiceRequest
    let len;
    let data = new findPlanServiceRequest(null);
    // Deserialize message field [pattern]
    data.pattern = std_msgs.msg.Int8MultiArray.deserialize(buffer, bufferOffset);
    // Deserialize message field [aesthetic_mode]
    data.aesthetic_mode = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Int8MultiArray.getMessageSize(object.pattern);
    return length + 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'sl_msgs/findPlanServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '861f24e62833b3969ce584a7874c91a6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int8MultiArray pattern
    std_msgs/Int8 aesthetic_mode
    
    ================================================================================
    MSG: std_msgs/Int8MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    int8[]            data          # array of data
    
    
    ================================================================================
    MSG: std_msgs/MultiArrayLayout
    # The multiarray declares a generic multi-dimensional array of a
    # particular data type.  Dimensions are ordered from outer most
    # to inner most.
    
    MultiArrayDimension[] dim # Array of dimension properties
    uint32 data_offset        # padding elements at front of data
    
    # Accessors should ALWAYS be written in terms of dimension stride
    # and specified outer-most dimension first.
    # 
    # multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
    #
    # A standard, 3-channel 640x480 image with interleaved color channels
    # would be specified as:
    #
    # dim[0].label  = "height"
    # dim[0].size   = 480
    # dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
    # dim[1].label  = "width"
    # dim[1].size   = 640
    # dim[1].stride = 3*640 = 1920
    # dim[2].label  = "channel"
    # dim[2].size   = 3
    # dim[2].stride = 3
    #
    # multiarray(i,j,k) refers to the ith row, jth column, and kth channel.
    
    ================================================================================
    MSG: std_msgs/MultiArrayDimension
    string label   # label of given dimension
    uint32 size    # size of given dimension (in type units)
    uint32 stride  # stride of given dimension
    ================================================================================
    MSG: std_msgs/Int8
    int8 data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new findPlanServiceRequest(null);
    if (msg.pattern !== undefined) {
      resolved.pattern = std_msgs.msg.Int8MultiArray.Resolve(msg.pattern)
    }
    else {
      resolved.pattern = new std_msgs.msg.Int8MultiArray()
    }

    if (msg.aesthetic_mode !== undefined) {
      resolved.aesthetic_mode = std_msgs.msg.Int8.Resolve(msg.aesthetic_mode)
    }
    else {
      resolved.aesthetic_mode = new std_msgs.msg.Int8()
    }

    return resolved;
    }
};

class findPlanServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.plan = null;
    }
    else {
      if (initObj.hasOwnProperty('plan')) {
        this.plan = initObj.plan
      }
      else {
        this.plan = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type findPlanServiceResponse
    // Serialize message field [plan]
    bufferOffset = std_msgs.msg.String.serialize(obj.plan, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type findPlanServiceResponse
    let len;
    let data = new findPlanServiceResponse(null);
    // Deserialize message field [plan]
    data.plan = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.plan);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'sl_msgs/findPlanServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '19fe59c3be17f06323e121cfc2b5607e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/String plan
    
    ================================================================================
    MSG: std_msgs/String
    string data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new findPlanServiceResponse(null);
    if (msg.plan !== undefined) {
      resolved.plan = std_msgs.msg.String.Resolve(msg.plan)
    }
    else {
      resolved.plan = new std_msgs.msg.String()
    }

    return resolved;
    }
};

module.exports = {
  Request: findPlanServiceRequest,
  Response: findPlanServiceResponse,
  md5sum() { return 'e752f9c85170cb73cc8cf790042cfd2d'; },
  datatype() { return 'sl_msgs/findPlanService'; }
};
