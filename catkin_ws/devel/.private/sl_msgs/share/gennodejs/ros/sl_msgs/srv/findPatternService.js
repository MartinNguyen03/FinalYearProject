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

class findPatternServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.num_eyelets = null;
      this.shoelace_length = null;
      this.eyelet_positions = null;
    }
    else {
      if (initObj.hasOwnProperty('num_eyelets')) {
        this.num_eyelets = initObj.num_eyelets
      }
      else {
        this.num_eyelets = new std_msgs.msg.Int8();
      }
      if (initObj.hasOwnProperty('shoelace_length')) {
        this.shoelace_length = initObj.shoelace_length
      }
      else {
        this.shoelace_length = new std_msgs.msg.Float32();
      }
      if (initObj.hasOwnProperty('eyelet_positions')) {
        this.eyelet_positions = initObj.eyelet_positions
      }
      else {
        this.eyelet_positions = new std_msgs.msg.Float32MultiArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type findPatternServiceRequest
    // Serialize message field [num_eyelets]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.num_eyelets, buffer, bufferOffset);
    // Serialize message field [shoelace_length]
    bufferOffset = std_msgs.msg.Float32.serialize(obj.shoelace_length, buffer, bufferOffset);
    // Serialize message field [eyelet_positions]
    bufferOffset = std_msgs.msg.Float32MultiArray.serialize(obj.eyelet_positions, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type findPatternServiceRequest
    let len;
    let data = new findPatternServiceRequest(null);
    // Deserialize message field [num_eyelets]
    data.num_eyelets = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    // Deserialize message field [shoelace_length]
    data.shoelace_length = std_msgs.msg.Float32.deserialize(buffer, bufferOffset);
    // Deserialize message field [eyelet_positions]
    data.eyelet_positions = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Float32MultiArray.getMessageSize(object.eyelet_positions);
    return length + 5;
  }

  static datatype() {
    // Returns string type for a service object
    return 'sl_msgs/findPatternServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '04424f9afefb944f67aec327d9aeeb55';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int8 num_eyelets
    std_msgs/Float32 shoelace_length
    std_msgs/Float32MultiArray eyelet_positions
    
    ================================================================================
    MSG: std_msgs/Int8
    int8 data
    
    ================================================================================
    MSG: std_msgs/Float32
    float32 data
    ================================================================================
    MSG: std_msgs/Float32MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    float32[]         data          # array of data
    
    
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
    const resolved = new findPatternServiceRequest(null);
    if (msg.num_eyelets !== undefined) {
      resolved.num_eyelets = std_msgs.msg.Int8.Resolve(msg.num_eyelets)
    }
    else {
      resolved.num_eyelets = new std_msgs.msg.Int8()
    }

    if (msg.shoelace_length !== undefined) {
      resolved.shoelace_length = std_msgs.msg.Float32.Resolve(msg.shoelace_length)
    }
    else {
      resolved.shoelace_length = new std_msgs.msg.Float32()
    }

    if (msg.eyelet_positions !== undefined) {
      resolved.eyelet_positions = std_msgs.msg.Float32MultiArray.Resolve(msg.eyelet_positions)
    }
    else {
      resolved.eyelet_positions = new std_msgs.msg.Float32MultiArray()
    }

    return resolved;
    }
};

class findPatternServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.pattern = null;
    }
    else {
      if (initObj.hasOwnProperty('pattern')) {
        this.pattern = initObj.pattern
      }
      else {
        this.pattern = new std_msgs.msg.Int8MultiArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type findPatternServiceResponse
    // Serialize message field [pattern]
    bufferOffset = std_msgs.msg.Int8MultiArray.serialize(obj.pattern, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type findPatternServiceResponse
    let len;
    let data = new findPatternServiceResponse(null);
    // Deserialize message field [pattern]
    data.pattern = std_msgs.msg.Int8MultiArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Int8MultiArray.getMessageSize(object.pattern);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'sl_msgs/findPatternServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd04d855767d2a79195175a4127d4dd3e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int8MultiArray pattern
    
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
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new findPatternServiceResponse(null);
    if (msg.pattern !== undefined) {
      resolved.pattern = std_msgs.msg.Int8MultiArray.Resolve(msg.pattern)
    }
    else {
      resolved.pattern = new std_msgs.msg.Int8MultiArray()
    }

    return resolved;
    }
};

module.exports = {
  Request: findPatternServiceRequest,
  Response: findPatternServiceResponse,
  md5sum() { return '145ffa40f05356208146cb6e34948f0c'; },
  datatype() { return 'sl_msgs/findPatternService'; }
};
