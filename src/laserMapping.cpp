#include "laser_mapping.h"
#include <csignal>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

bool FLAG_EXIT = false;
auto laser_mapping = make_shared<LaserMapping>();       // ikdtree中存在多线程，所以必须定义为全局变量 https://github.com/hku-mars/ikd-Tree/issues/8
void SigHandle(int sig){
    FLAG_EXIT = true;
    ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();

    laser_mapping->InitROS(nh);
    
    while(status){
        if(FLAG_EXIT) break;
        ros::spinOnce();

        laser_mapping->Run();
        rate.sleep();
    }

    laser_mapping->Finish();
}