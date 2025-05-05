#include "behaviortree_cpp_v3/bt_factory.h"
#include "behaviortree_cpp_v3/action_node.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "rosllm_srvs/DetectRope.h"
#include "rosllm_srvs/DetectTarget.h"
#include "rosllm_srvs/ObserveScene.h"
#include "rosllm_srvs/Gripper.h"
#include "rosllm_srvs/HandOver.h"
#include "rosllm_srvs/InsertObject.h"
#include "rosllm_srvs/MoveArm.h"
#include "rosllm_srvs/MoveDefault.h"
#include "rosllm_srvs/PullObject.h"

// Function to set input from the behavior tree
void setInput(const std_msgs::String& name, const BT::InputPort& value);

class DetectRopeAction : public BT::SyncActionNode {
    public:
        DetectRopeAction(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config), client(nh.serviceClient<rosllm_srvs::DetectRope>("detect_object")) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std_msgs::String>("object_name"),
                BT::OutputPort<geometry_msgs::PoseArray>("object_pose") 
            };
        }
    
        BT::NodeStatus tick() override {
            rosllm_srvs::DetectRope srv;
            
            // Get object name from input port
            if (!getInput("object_name", srv.request.object_name)) {
                ROS_ERROR("DetectRope: Missing object_name input.");
                return BT::NodeStatus::FAILURE;
            }
    
            if (client.call(srv) && srv.response.success) {
                setOutput("object_pose", srv.response.object_pose);
                ROS_INFO("Object '%s' detected.", srv.request.object_name.data.c_str());
                return BT::NodeStatus::SUCCESS;
            }
    
            ROS_WARN("Object '%s' not detected.", srv.request.object_name.data.c_str());
            return BT::NodeStatus::FAILURE;
        }
    
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
};

class DetectTargetAction : public BT::SyncActionNode {
    public:
        DetectTargetAction(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config), client(nh.serviceClient<rosllm_srvs::DetectTarget>("detect_target")) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std_msgs::String>("target_name"),  // Input for target name
                BT::OutputPort<geometry_msgs::PoseArray>("target_pose")  // Output for detected pose
            };
        }
    
        BT::NodeStatus tick() override {
            rosllm_srvs::DetectTarget srv;
            
            // Get target name from input port
            if (!getInput("target_name", srv.request.target_name)) {
                ROS_ERROR("DetectTarget: Missing target_name input.");
                return BT::NodeStatus::FAILURE;
            }
    
            if (client.call(srv) && srv.response.success) {
                setOutput("target_pose", srv.response.target_pose);
                ROS_INFO("Target '%s' detected.", srv.request.target_name.data.c_str());
                return BT::NodeStatus::SUCCESS;
            }
    
            ROS_WARN("Target '%s' not detected.", srv.request.target_name.data.c_str());
            return BT::NodeStatus::FAILURE;
        }
    
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
};

class MoveArmAction : public BT::SyncActionNode {
    public:
        MoveArmAction(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config), client(nh.serviceClient<rosllm_srvs::MoveArm>("move_arm")) {}
    
        static BT::PortsList providedPorts() {
            return {
                BT::InputPort<std_msgs::String>("arm"),
                BT::InputPort<geometry_msgs::Pose>("target_pose")
            };
        }
    
        BT::NodeStatus tick() override {
            rosllm_srvs::MoveArm srv;
            if (!getInput("arm", srv.request.arm) || !getInput("target_pose", srv.request.target_pose)) {
                ROS_ERROR("Missing input parameters.");
                return BT::NodeStatus::FAILURE;
            }
            if (client.call(srv) && srv.response.success) {
                ROS_INFO("Arm moved successfully.");
                return BT::NodeStatus::SUCCESS;
            }
            return BT::NodeStatus::FAILURE;
        }
    
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
};
        
class MoveToDefaultAction : public MoveArmAction {
    public:
        MoveToDefaultAction(const std::string& name, const BT::NodeConfiguration& config)
            : MoveArmAction(name, config) {}
    
        static BT::PortsList providedPorts() {
            return { BT::InputPort<std_msgs::String>("arm") };  // "left", "right", or "both"
        }
    
        BT::NodeStatus tick() override {
            std_msgs::String arm;
            if (!getInput("arm", arm)) {
                ROS_ERROR("No arm specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            geometry_msgs::Pose left_default = {0.3, 0.2, 0.5};  // Example default for left arm
            geometry_msgs::Pose right_default = {0.3, -0.2, 0.5}; // Example default for right arm
    
            if (arm == "left") {
                setInput("target_pose", left_default);
                return MoveArmAction::tick();
            } 
            else if (arm == "right") {
                setInput("target_pose", right_default);
                return MoveArmAction::tick();
            } 
            else if (arm == "both") {
                // Move left arm first, then right arm
                setInput("target_pose", left_default);
                if (MoveArmAction::tick() == BT::NodeStatus::FAILURE) {
                    return BT::NodeStatus::FAILURE;
                }
    
                setInput("target_pose", right_default);
                if (MoveArmAction::tick() == BT::NodeStatus::FAILURE) {
                    return BT::NodeStatus::FAILURE;
                }
            } 
            else {
                ROS_ERROR("Invalid arm specified: %s", arm.data.c_str());
                return BT::NodeStatus::FAILURE;
            }
        }
};

class InsertObjectAction : public MoveArmAction {
    public:
        InsertObjectAction(const std::string& name, const BT::NodeConfiguration& config)
            : MoveArmAction(name, config) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std_msgs::String>("arm"),  // "left" or "right"
                BT::InputPort<geometry_msgs::Pose>("target_pose")  // pose for insertion
            };
        }
    
        BT::NodeStatus tick() override {
            std_msgs::String arm;
            geometry_msgs::Pose target_pose;
    
            if (!getInput("arm", arm)) {
                ROS_ERROR("No arm specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            if (!getInput("target_pose", target_pose)) {
                ROS_ERROR("No insertion pose specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            setInput("target_pose", target_pose);
            return MoveArmAction::tick();
        }

    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
};

class PullObjectAction : public MoveArmAction {
    public:
        PullObjectAction(const std::string& name, const BT::NodeConfiguration& config)
            : MoveArmAction(name, config), client(nh.serviceClient<rosllm_srvs::PullObject>("pull_object")) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std_msgs::String>("arm"),  // "left" or "right"
                BT::InputPort<geometry_msgs::Pose>("target_pose")  // Target pose to pull the object
            };
        }
    
        BT::NodeStatus tick() override {
            std_msgs::String arm;
            geometry_msgs::Pose pull_pose;
    
            if (!getInput("arm", arm)) {
                ROS_ERROR("No arm specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            if (!getInput("pull_pose", pull_pose)) {
                ROS_ERROR("No pull pose specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            setInput("target_pose", pull_pose);
            return MoveArmAction::tick();
        }
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
};
    
    
    
class GripperAction : public BT::SyncActionNode {
    public:
        GripperAction(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config), client(nh.serviceClient<rosllm_srvs::Gripper>("gripper")) {}
    
        static BT::PortsList providedPorts() {
            return {
                BT::InputPort<std_msgs::String>("action"),
                BT::InputPort<std_msgs::String>("arm")
            };
        }
    
        BT::NodeStatus tick() override {
            rosllm_srvs::Gripper srv;
            if (!getInput("action", srv.request.action) || !getInput("arm", srv.request.arm)) {
                ROS_ERROR("Missing gripper parameters.");
                return BT::NodeStatus::FAILURE;
            }
            if (srv.request.action.data != "open" && srv.request.action.data != "close") {
                ROS_ERROR("Invalid gripper action: %s", srv.request.action.data.c_str());
                return BT::NodeStatus::FAILURE;
            }
            if (srv.request.arm.data != "left" && srv.request.arm.data != "right") {
                ROS_ERROR("Invalid arm specified: %s", srv.request.arm.data.c_str());
                return BT::NodeStatus::FAILURE;
            }
            if (client.call(srv) && srv.response.success) {
                ROS_INFO("Gripper action executed.");
                return BT::NodeStatus::SUCCESS;
            }
            return BT::NodeStatus::FAILURE;
        }
    
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
};

class HandoverAction : public BT::SyncActionNode {
public:
    HandoverAction(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), client(nh.serviceClient<rosllm_srvs::HandOver>("hand_over")) {}

    static BT::PortsList providedPorts() {
        return {
            BT::InputPort<std_msgs::String>("from_arm"),
            BT::InputPort<std_msgs::String>("to_arm")
        };
    }

    BT::NodeStatus tick() override {
        rosllm_srvs::HandOver srv;
        if (!getInput("from_arm", srv.request.from_arm) || !getInput("to_arm", srv.request.to_arm)) {
            ROS_ERROR("Missing handover parameters.");
            return BT::NodeStatus::FAILURE;
        }
        if (srv.request.from_arm.data == srv.request.to_arm.data) {
            ROS_ERROR("Cannot hand over to the same arm.");
            return BT::NodeStatus::FAILURE;
        }
        if (client.call(srv) && srv.response.success) {
            ROS_INFO("Handover action executed.");
            return BT::NodeStatus::SUCCESS; 
        }
        return BT::NodeStatus::FAILURE;
    }

private:
    ros::NodeHandle nh;
    ros::ServiceClient client;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "yumi_behavior_tree");
    ros::NodeHandle nh;

    BT::BehaviorTreeFactory factory;

    factory.registerNodeType<DetectRopeAction>("DetectRope");
    factory.registerNodeType<GripperAction>("GripperAction");
    factory.registerNodeType<HandoverAction>("HandoverAction");

    auto tree = factory.createTreeFromFile("/path/to/your/tree.xml");

    ros::Rate rate(10);
    while (ros::ok()) {
        tree.tickRoot();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
