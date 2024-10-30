#ifndef GNSS_PROCESSING_H
#define GNSS_PROCESSING_H

#include "earth.h"
#include "common_lib.h"
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <ceres/ceres.h>

class GNSSProcessing
{
public:
    using Ptr = std::shared_ptr<GNSSProcessing>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GNSSProcessing();
    ~GNSSProcessing();

    void input_gnss(const sensor_msgs::NavSatFixConstPtr& gnss);

    void input_imu(const deque<sensor_msgs::Imu::ConstPtr>& imu);

    bool Initialization(Vector3d &mean_acc, Vector3d &mean_gyr, Vector3d &cov_acc, Vector3d &cov_gyr, bool is_eular);

    void process(const vector<Pose6D>& imu_pos, const double pcl_beg_time);
    
private:
    Vector3d    anchor_;       // 锚点
    bool        is_origin_set;     // 是否已经设置了锚点
    Vector3d    eular_;

    GNSS    gnss_;
    GNSS    last_gnss_;

    deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
    bool    is_has_zero_velocity;

    bool detectZeroVelocity(Vector3d &mean_acc, Vector3d &mean_gyr, Vector3d &cov_acc, Vector3d &cov_gyr);

    void optimize();
};

#endif