// Auto-generated. Do not edit!

// (in-package uni_lace_msgs.srv)


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

class UniLaceStepServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.act = null;
    }
    else {
      if (initObj.hasOwnProperty('act')) {
        this.act = initObj.act
      }
      else {
        this.act = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UniLaceStepServiceRequest
    // Serialize message field [act]
    bufferOffset = std_msgs.msg.String.serialize(obj.act, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UniLaceStepServiceRequest
    let len;
    let data = new UniLaceStepServiceRequest(null);
    // Deserialize message field [act]
    data.act = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.act);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UniLaceStepServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '09db63d1b2a62392d56e43d16c36ff60';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #request
    std_msgs/String act
    
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
    const resolved = new UniLaceStepServiceRequest(null);
    if (msg.act !== undefined) {
      resolved.act = std_msgs.msg.String.Resolve(msg.act)
    }
    else {
      resolved.act = new std_msgs.msg.String()
    }

    return resolved;
    }
};

class UniLaceStepServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.obs = null;
      this.obs_raw = null;
    }
    else {
      if (initObj.hasOwnProperty('obs')) {
        this.obs = initObj.obs
      }
      else {
        this.obs = new std_msgs.msg.String();
      }
      if (initObj.hasOwnProperty('obs_raw')) {
        this.obs_raw = initObj.obs_raw
      }
      else {
        this.obs_raw = new std_msgs.msg.UInt8MultiArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UniLaceStepServiceResponse
    // Serialize message field [obs]
    bufferOffset = std_msgs.msg.String.serialize(obj.obs, buffer, bufferOffset);
    // Serialize message field [obs_raw]
    bufferOffset = std_msgs.msg.UInt8MultiArray.serialize(obj.obs_raw, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UniLaceStepServiceResponse
    let len;
    let data = new UniLaceStepServiceResponse(null);
    // Deserialize message field [obs]
    data.obs = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    // Deserialize message field [obs_raw]
    data.obs_raw = std_msgs.msg.UInt8MultiArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.obs);
    length += std_msgs.msg.UInt8MultiArray.getMessageSize(object.obs_raw);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UniLaceStepServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd213a8c9c9c20d75e8f685fc69fa73a6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #response
    std_msgs/String obs
    std_msgs/UInt8MultiArray obs_raw
    
    ================================================================================
    MSG: std_msgs/String
    string data
    
    ================================================================================
    MSG: std_msgs/UInt8MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    uint8[]           data          # array of data
    
    
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
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new UniLaceStepServiceResponse(null);
    if (msg.obs !== undefined) {
      resolved.obs = std_msgs.msg.String.Resolve(msg.obs)
    }
    else {
      resolved.obs = new std_msgs.msg.String()
    }

    if (msg.obs_raw !== undefined) {
      resolved.obs_raw = std_msgs.msg.UInt8MultiArray.Resolve(msg.obs_raw)
    }
    else {
      resolved.obs_raw = new std_msgs.msg.UInt8MultiArray()
    }

    return resolved;
    }
};

module.exports = {
  Request: UniLaceStepServiceRequest,
  Response: UniLaceStepServiceResponse,
  md5sum() { return 'af9f61a907e31117188b236c02aaf4bd'; },
  datatype() { return 'uni_lace_msgs/UniLaceStepService'; }
};
