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

class UniLaceInfoServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.render = null;
    }
    else {
      if (initObj.hasOwnProperty('render')) {
        this.render = initObj.render
      }
      else {
        this.render = new std_msgs.msg.Bool();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UniLaceInfoServiceRequest
    // Serialize message field [render]
    bufferOffset = std_msgs.msg.Bool.serialize(obj.render, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UniLaceInfoServiceRequest
    let len;
    let data = new UniLaceInfoServiceRequest(null);
    // Deserialize message field [render]
    data.render = std_msgs.msg.Bool.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UniLaceInfoServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b38f785d3e981b63eca9d66d30fbd88a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #request
    std_msgs/Bool render
    
    ================================================================================
    MSG: std_msgs/Bool
    bool data
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new UniLaceInfoServiceRequest(null);
    if (msg.render !== undefined) {
      resolved.render = std_msgs.msg.Bool.Resolve(msg.render)
    }
    else {
      resolved.render = new std_msgs.msg.Bool()
    }

    return resolved;
    }
};

class UniLaceInfoServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.info = null;
    }
    else {
      if (initObj.hasOwnProperty('info')) {
        this.info = initObj.info
      }
      else {
        this.info = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UniLaceInfoServiceResponse
    // Serialize message field [info]
    bufferOffset = std_msgs.msg.String.serialize(obj.info, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UniLaceInfoServiceResponse
    let len;
    let data = new UniLaceInfoServiceResponse(null);
    // Deserialize message field [info]
    data.info = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.info);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UniLaceInfoServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '12a155e3803bd9637acdd5f5509da193';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #response
    std_msgs/String info
    
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
    const resolved = new UniLaceInfoServiceResponse(null);
    if (msg.info !== undefined) {
      resolved.info = std_msgs.msg.String.Resolve(msg.info)
    }
    else {
      resolved.info = new std_msgs.msg.String()
    }

    return resolved;
    }
};

module.exports = {
  Request: UniLaceInfoServiceRequest,
  Response: UniLaceInfoServiceResponse,
  md5sum() { return '3a92e7221553a608fa193b6ae18fdde5'; },
  datatype() { return 'uni_lace_msgs/UniLaceInfoService'; }
};
