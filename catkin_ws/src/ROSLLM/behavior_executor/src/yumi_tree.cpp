#include "behaviortree_cpp_v3/bt_factory.h"
#include "behaviortree_cpp_v3/action_node.h"
#include "ros/ros.h"
#include "std_srvs/Trigger.h"


class DetectObjectAction : public BT::SyncActionNode {
public:
    DetectObjectAction(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), client(nh.serviceClient<std_srvs::Trigger>("object_name")) {}

    static BT::PortsList providedPorts() {
        return { BT::OutputPort<std::vector<float>>("object_location") };
    }

    BT::NodeStatus tick() override {
        std_srvs::Trigger srv;
        if (client.call(srv) && srv.response.success) {
            setOutput("object_location", objectLocation);
            ROS_INFO("Object Detected at [%f, %f, %f]", objectLocation[0], objectLocation[1], objectLocation[2]);
            return BT::NodeStatus::SUCCESS;
        }
        return BT::NodeStatus::FAILURE;
    }

private:
    ros::NodeHandle nh;
    ros::ServiceClient client;
};

class DetectTargetAction : public BT::SyncActionNode {
    public:
        DetectTargetAction(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config), client(nh.serviceClient<std_srvs::Trigger>("target_name")) {}
    
        static BT::PortsList providedPorts() {
            return { BT::OutputPort<std::vector<float>>("target_location") };
        }
    
        BT::NodeStatus tick() override {
            std_srvs::Trigger srv;
            if (client.call(srv) && srv.response.success) {
                setOutput("target_location", targetLocation);
                ROS_INFO("Target Detected at [%f, %f, %f]", targetLocation[0], targetLocation[1], targetLocation[2]);
                return BT::NodeStatus::SUCCESS;
            }
            return BT::NodeStatus::FAILURE;
        }
    
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client;
    };

class MoveToAction : public BT::SyncActionNode {
    public:
        MoveToAction(const std::string& name, const BT::NodeConfiguration& config)
            : BT::SyncActionNode(name, config),
              client_left(nh.serviceClient<std_srvs::Trigger>("move_to_left")),
              client_right(nh.serviceClient<std_srvs::Trigger>("move_to_right")) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std::vector<float>>("target_location"),
                BT::InputPort<std::string>("arm")  // "left" or "right" only
            };
        }
    
        BT::NodeStatus tick() override {
            std::vector<float> target_location;
            std::string arm;
            
            if (!getInput("target_location", target_location)) {
                return BT::NodeStatus::FAILURE;
            }
            if (!getInput("arm", arm) || (arm != "left" && arm != "right")) {
                ROS_ERROR("Invalid or missing arm input. Use 'left' or 'right'.");
                return BT::NodeStatus::FAILURE;
            }
    
            std_srvs::Trigger srv;
            if (arm == "left") {
                if (client_left.call(srv) && srv.response.success) {
                    ROS_INFO("Left arm moved to [%f, %f, %f]", target_location[0], target_location[1], target_location[2]);
                    return BT::NodeStatus::SUCCESS;
                }
            } else { // Must be "right"
                if (client_right.call(srv) && srv.response.success) {
                    ROS_INFO("Right arm moved to [%f, %f, %f]", target_location[0], target_location[1], target_location[2]);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            return BT::NodeStatus::FAILURE;
        }
    
    private:
        ros::NodeHandle nh;
        ros::ServiceClient client_left;
        ros::ServiceClient client_right;
};
        
class MoveToDefaultAction : public MoveToAction {
    public:
        MoveToDefaultAction(const std::string& name, const BT::NodeConfiguration& config)
            : MoveToAction(name, config) {}
    
        static BT::PortsList providedPorts() {
            return { BT::InputPort<std::string>("arm") };  // "left", "right", or "both"
        }
    
        BT::NodeStatus tick() override {
            std::string arm;
            if (!getInput("arm", arm)) {
                ROS_ERROR("No arm specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            std::vector<float> left_default = {0.3, 0.2, 0.5};  // Example default for left arm
            std::vector<float> right_default = {0.3, -0.2, 0.5}; // Example default for right arm
    
            if (arm == "left") {
                setInput("target_location", left_default);
                return MoveToAction::tick();
            } 
            else if (arm == "right") {
                setInput("target_location", right_default);
                return MoveToAction::tick();
            } 
            else if (arm == "both") {
                // Move left arm first, then right arm
                setInput("target_location", left_default);
                if (MoveToAction::tick() == BT::NodeStatus::FAILURE) {
                    return BT::NodeStatus::FAILURE;
                }
    
                setInput("target_location", right_default);
                return MoveToAction::tick();
            } 
            else {
                ROS_ERROR("Invalid arm specified: %s", arm.c_str());
                return BT::NodeStatus::FAILURE;
            }
        }
};

class InsertObjectAction : public MoveToAction {
    public:
        InsertObjectAction(const std::string& name, const BT::NodeConfiguration& config)
            : MoveToAction(name, config) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std::string>("arm"),  // "left" or "right"
                BT::InputPort<std::vector<float>>("insert_position")  // Position for insertion
            };
        }
    
        BT::NodeStatus tick() override {
            std::string arm;
            std::vector<float> insert_position;
    
            if (!getInput("arm", arm)) {
                ROS_ERROR("No arm specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            if (!getInput("insert_position", insert_position)) {
                ROS_ERROR("No insertion position specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            setInput("target_location", insert_position);
            return MoveToAction::tick();
        }
};

class PullObjectAction : public MoveToAction {
    public:
        PullObjectAction(const std::string& name, const BT::NodeConfiguration& config)
            : MoveToAction(name, config) {}
    
        static BT::PortsList providedPorts() {
            return { 
                BT::InputPort<std::string>("arm"),  // "left" or "right"
                BT::InputPort<std::vector<float>>("pull_position")  // Target position to pull the object
            };
        }
    
        BT::NodeStatus tick() override {
            std::string arm;
            std::vector<float> pull_position;
    
            if (!getInput("arm", arm)) {
                ROS_ERROR("No arm specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            if (!getInput("pull_position", pull_position)) {
                ROS_ERROR("No pull position specified.");
                return BT::NodeStatus::FAILURE;
            }
    
            setInput("target_location", pull_position);
            return MoveToAction::tick();
        }
};
    
    
    
class GripperAction : public BT::SyncActionNode {
public:
    GripperAction(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config),
          client_open(nh.serviceClient<std_srvs::Trigger>("gripper_open")),
          client_close(nh.serviceClient<std_srvs::Trigger>("gripper_close")) {}

    static BT::PortsList providedPorts() {
        return { BT::InputPort<std::string>("action") };
    }

    BT::NodeStatus tick() override {
        std::string action;
        if (!getInput("action", action)) {
            return BT::NodeStatus::FAILURE;
        }

        std_srvs::Trigger srv;
        if ((action == "open" && client_open.call(srv) && srv.response.success) ||
            (action == "close" && client_close.call(srv) && srv.response.success)) {
            ROS_INFO("Gripper %s successful", action.c_str());
            return BT::NodeStatus::SUCCESS;
        }
        return BT::NodeStatus::FAILURE;
    }

private:
    ros::NodeHandle nh;
    ros::ServiceClient client_open;
    ros::ServiceClient client_close;
    ros::ServiceClient client_left;
    ros::ServiceClient client_right;
};

class HandoverAction : public BT::SyncActionNode {
public:
    HandoverAction(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config),
          client_left(nh.serviceClient<std_srvs::Trigger>("handover_left")),
          client_right(nh.serviceClient<std_srvs::Trigger>("handover_right")) {}

    static BT::PortsList providedPorts() {
        return {};
    }

    BT::NodeStatus tick() override {
        std_srvs::Trigger srv;
        return (client_left.call(srv) && client_right.call(srv) && srv.response.success) ?
            BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }

private:
    ros::NodeHandle nh;
    ros::ServiceClient client_left;
    ros::ServiceClient client_right;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "yumi_behavior_tree");
    ros::NodeHandle nh;

    BT::BehaviorTreeFactory factory;

    factory.registerNodeType<DetectObjectAction>("DetectObject");
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
