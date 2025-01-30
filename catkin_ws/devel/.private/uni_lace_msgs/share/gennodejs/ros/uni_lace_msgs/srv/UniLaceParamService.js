// Auto-generated. Do not edit!

// (in-package uni_lace_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class UniLaceParamServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UniLaceParamServiceRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UniLaceParamServiceRequest
    let len;
    let data = new UniLaceParamServiceRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UniLaceParamServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #request
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new UniLaceParamServiceRequest(null);
    return resolved;
    }
};

class UniLaceParamServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.params_json = null;
    }
    else {
      if (initObj.hasOwnProperty('params_json')) {
        this.params_json = initObj.params_json
      }
      else {
        this.params_json = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UniLaceParamServiceResponse
    // Serialize message field [params_json]
    bufferOffset = std_msgs.msg.String.serialize(obj.params_json, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UniLaceParamServiceResponse
    let len;
    let data = new UniLaceParamServiceResponse(null);
    // Deserialize message field [params_json]
    data.params_json = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.params_json);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'uni_lace_msgs/UniLaceParamServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'dd5debcaac10dc60355cf969fc28100a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    #response
    std_msgs/String params_json
    
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
    const resolved = new UniLaceParamServiceResponse(null);
    if (msg.params_json !== undefined) {
      resolved.params_json = std_msgs.msg.String.Resolve(msg.params_json)
    }
    else {
      resolved.params_json = new std_msgs.msg.String()
    }

    return resolved;
    }
};

module.exports = {
  Request: UniLaceParamServiceRequest,
  Response: UniLaceParamServiceResponse,
  md5sum() { return 'dd5debcaac10dc60355cf969fc28100a'; },
  datatype() { return 'uni_lace_msgs/UniLaceParamService'; }
};
