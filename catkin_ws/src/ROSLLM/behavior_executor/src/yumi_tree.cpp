#include <behaviortree_ros/bt_service_node.h>
#include <behaviortree_ros/bt_action_node.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <rosllm_srvs/ExecuteBehaviour.h>
#include <rosllm_srvs/ExecuteBehaviourRequest.h>

using namespace BT;

class VLMCheck : public RosServiceNode<rosllm_srvs::VLM>
{
public:
    VLMCheck(ros::NodeHandle& handle, const std::string& node_name, const NodeConfiguration & conf)
        : RosServiceNode<rosllm_srvs::VLM>(handle, node_name, conf) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<sensor_msgs::Image>("img"),
        };
    }

    void sendRequest(RequestType& request) override
    {
        request.prompt = "Here is the updated scene, would you like to proceed; respond with yes or no";
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
    void halt() override
    {
        if (status() == NodeStatus::RUNNING)
        {
            ROS_WARN("YuMi Action halted");
            BaseClass::halt();
        }
    }

};

class YumiNode : public RosServiceNode<rosllm_srvs::ExecuteBehaviour>
{
public:
    YumiNode(ros::NodeHandle& handle, const std::string& node_name, const NodeConfiguration & conf)
        : RosServiceNode<rosllm_srvs::ExecuteBehaviour>(handle, node_name, conf) {}

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
        ROS_INFO("YuMi Action: Executed");
        if (rep.success)
        {
            setOutput<string>("message", rep.task);
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
    void halt() override
    {
        if (status() == NodeStatus::RUNNING)
        {
            ROS_WARN("YuMi Action halted");
            BaseClass::halt();
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bt_executor_node");
    ros::NodeHandle nh;

    BehaviorTreeFactory factory;

    ROS_INFO("Registering BT nodes...");
    factory.registerNodeType<YumiNode>("YumiNode");

    ROS_INFO("Starting BT executor service...");
    ros::spin();
    return 0;
}