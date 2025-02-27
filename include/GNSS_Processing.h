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

#include "common_lib.h"
#include "earth.h"
#include "Factors.h"
#include <ceres/ceres.h>
#include <fstream>

class GNSSProcessing{
public:
    using Ptr = std::shared_ptr<GNSSProcessing>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GNSSProcessing(ros::NodeHandle& nh);
    ~GNSSProcessing();
 
    void readrtkresult(const string gnss_path);
    
    void addIMUpos(const vector<Pose6D> &IMUpose, const double pcl_beg_time);

    void computeH(Eigen::MatrixXd &HTH, Eigen::MatrixXd &HTL, Eigen::Matrix3d &rot_end, Eigen::Vector3d pos_end);

    bool    new_gnss_;

    StatesGroup *state;             // 后验值
    StatesGroup *state_propagat;    // 先验值

private:
    void optimize();

    void Initialization();

    std::queue<GNSS>    gnss_queue_;
    std::vector<GNSS>   gnss_path_;
    std::vector<V3D>    odo_path_;
    std::vector<M3D>    odo_rot_;

    bool    is_init_;
    bool    is_anchor_;
    V3D     anchor_;
    V3D     antlever_;                // 杆臂 Tia 右前上坐标系到gnss接收机天线
    M3D     rot_enu2global_;

    V3D     delta_pos_;               // gnss接收时刻imu位姿到当前帧结束时刻imu位姿的变换
    M3D     delta_rot_;               // R_gnssi_endi
    GNSS    gnss_;
    GNSS    last_gnss_;
    double  last_gnss_time_;
};
#endif