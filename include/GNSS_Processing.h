/*
 * @file: 
 * @brief:  
 * @details: 
 * @author: Xueqing Zhang. Email: zxq@whu.edu.cn
 * @date : Do not edit
 * @version: 
 * @par: Copyright(c) 2012-2021 School of Geodesy and Geomatics, University of Wuhan. All Rights Reserved. 
 * POSMind Software Kernel, named IPS (Inertial Plus Sensing), this file is a part of IPS.
 */
#ifndef GNSS_PROCESSING_H
#define GNSS_PROCESSING_H

#include "earth.h"
#include "common_lib.h"
#include "Factors.h"
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <ceres/ceres.h>
#include <mutex>
#include <thread>

class GNSSProcessing
{
public:
    using Ptr = std::shared_ptr<GNSSProcessing>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GNSSProcessing(ros::NodeHandle& nh);
    ~GNSSProcessing();

    void input_gnss(const sensor_msgs::NavSatFix::ConstPtr& msg_in);

    void input_path(const nav_msgs::Odometry::ConstPtr& msg_in);

    void Initialization();

    bool optimize();
    
private:
    Vector3d            anchor_;        // 锚点，弧度制
    bool                is_origin_set;  // 是否已经设置了锚点
    bool                is_has_yaw_;    // 是否有yaw角
    double              yaw_;           // gnss to odo的航偏角
    Vector3d            antlever_;      // 杆臂

    bool                new_gnss_;
    std::thread         thread_opt_;

    ros::Subscriber odo_sub_;
    ros::Subscriber gnss_sub_;
    std::map<double, Vector3d> odo_path_;
    std::map<double, GNSS> gnss_buffer_;
    double last_gnss_time_;
    std::queue<GNSS> gnss_queue_;

    mutex odo_mutex_;
    mutex gnss_mutex_;

    void gnssOutlierCullingByChi2(ceres::Problem & problem,
                            vector<std::pair<ceres::ResidualBlockId, GNSS>> &residual_block);
};

#endif