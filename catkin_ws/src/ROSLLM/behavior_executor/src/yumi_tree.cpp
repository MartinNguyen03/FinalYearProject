#include <behaviortree_ros/bt_service_node.h>
#include <behaviortree_ros/bt_action_node.h>
#include <behaviortree_cpp_v3/loggers/bt_zmq_publisher.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <rosllm_srvs/VLM.h>
#include <rosllm_srvs/VLMRequest.h>
#include <rosllm_srvs/ExecuteBehavior.h>
#include <rosllm_srvs/ExecuteBehaviorRequest.h>
#include <fstream>
#include <ros/package.h>

using namespace BT;
class PrintValue : public BT::SyncActionNode
{
public:
  PrintValue(const std::string& name, const BT::NodeConfiguration& config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override {
    int value = 0;
    if( getInput("message", value ) ){
      std::cout << "PrintValue: " << value << std::endl;
      return NodeStatus::SUCCESS;
    }
    else{
      std::cout << "PrintValue FAILED "<< std::endl;
      return NodeStatus::FAILURE;
    }
  }

  static BT::PortsList providedPorts() {
    return{ BT::InputPort<int>("message") };
  }
};

class VisualCheck : public RosServiceNode<rosllm_srvs::VLM>
{
public:
    VisualCheck(ros::NodeHandle& handle, const std::string& node_name, const NodeConfiguration & conf)
        : RosServiceNode<rosllm_srvs::VLM>(handle, node_name, conf) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<sensor_msgs::Image>("img"),
            OutputPort<std::string>("response")
        };
    }

    void sendRequest(RequestType& request) override
    {
        request.prompt = "Here is the updated scene, would you like to proceed; respond with 'yes' or 'no'";
        getInput("img", request.img);
    }

    NodeStatus onResponse(const ResponseType& rep) override
    {
        ROS_INFO("VLM Check: Executed");
        if (rep.response == "yes")
        {
            ROS_INFO("VLM Check: Proceeding with the task");
            return NodeStatus::SUCCESS;
        }
        else
        {
            ROS_ERROR("VLM Check: Failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailedRequest(FailureCause failure) override
    {
        ROS_ERROR("YuMi Action request failed %d", static_cast<int>(failure));
        return NodeStatus::FAILURE;
    }
};

class YumiAction : public RosServiceNode<rosllm_srvs::ExecuteBehavior>
{
public:
    YumiAction(ros::NodeHandle& handle, const std::string& node_name, const NodeConfiguration & conf)
        : RosServiceNode<rosllm_srvs::ExecuteBehavior>(handle, node_name, conf) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<std::string>("action"),
            InputPort<std::string>("rope"),
            InputPort<std::string>("marker"),
            InputPort<std::string>("site"),
            OutputPort<std::string>("message")
        };
    }

    void sendRequest(RequestType& request) override
    {
        getInput("action", request.action);
        getInput("rope", request.rope);
        getInput("marker", request.marker);
        getInput("site", request.site);
    }

    NodeStatus onResponse(const ResponseType& rep) override
    {
        ROS_INFO("YuMi Action: Executed - %s", rep.description.c_str());
        ros::Duration(1.0).sleep(); // Simulate some processing time
        if (rep.success)
        {
            setOutput<std::string>("message", rep.description);
            return NodeStatus::SUCCESS;
        }
        else
        {
            ROS_ERROR("YuMi Action: Failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailedRequest(FailureCause failure) override
    {
        ROS_ERROR("YuMi Action request failed %d", static_cast<int>(failure));
        return NodeStatus::FAILURE;
    }

};

int main(int argc, char** argv )
{
    ros::init(argc, argv, "bt_executor_node");
    ros::NodeHandle nh;

    BehaviorTreeFactory factory;

    ROS_INFO("Registering BT nodes...");
    // factory.registerNodeType<PrintValue>("PrintValue");
    RegisterRosService<YumiAction>(factory, "YumiAction", nh);
    RegisterRosService<VisualCheck>(factory, "VisualCheck", nh);
    ROS_INFO("Starting BT executor service...");
    ROS_INFO("BT nodes registered successfully.");
    // Read the XML file into a string
    ROS_INFO("Reading XML file...");
    std::string pkg_path = ros::package::getPath("behavior_executor");
    std::string xml_file = pkg_path + "/config/gen_tree.xml";
    auto tree = factory.createTreeFromFile(xml_file);
    ROS_INFO("Behavior Tree created successfully.");
    ROS_INFO("Starting Behavior Tree execution...");
    NodeStatus status = NodeStatus::IDLE;

    ROS_INFO("Starting BT executor...");
    while( ros::ok() && (status == NodeStatus::IDLE || status == NodeStatus::RUNNING))
      {
        ros::spinOnce();
        status = tree.tickRoot();
        std::cout << status << std::endl;
        ros::Duration sleep_time(0.01);
        sleep_time.sleep();
      }

      return 0;
    }