#include "GNSS_Processing.h"
#include <fstream>

#define ZERO_VELOCITY_ACC_THRESHOLD 0.1    // 零速阈值
#define ZERO_VELOCITY_GYR_THRESHOLD 0.1
#define MINMUM_ALIGN_VELOCITY 0.5

GNSSProcessing::GNSSProcessing()
{
    is_origin_set = false;
    is_has_zero_velocity = false;
    eular_ = Vector3d(0, 0, 0);
    anchor_ = Vector3d(0, 0, 0);
    imu_buffer_.clear();
}

GNSSProcessing::~GNSSProcessing(){ }

void GNSSProcessing::input_gnss(const sensor_msgs::NavSatFixConstPtr& msg_in)
{
    if(!is_origin_set){
        anchor_ = Vector3d(msg_in->latitude, msg_in->longitude, msg_in->altitude);
        // G_m_s2  = earth::gravity(anchor_);
        is_origin_set = true;
    }

    GNSS gnss;
    gnss.time = msg_in->header.stamp.toSec();
    gnss.blh  = Earth::global2local(anchor_, Vector3d(msg_in->latitude, msg_in->longitude, msg_in->altitude));
    gnss.std  = Vector3d(msg_in->position_covariance[0], msg_in->position_covariance[4], msg_in->position_covariance[8]);
    last_gnss_= gnss_;
    gnss_     = gnss;
}

void GNSSProcessing::input_imu(const deque<sensor_msgs::Imu::ConstPtr>& imu)
{
    for(auto iter:imu){
        imu_buffer_.push_back(iter);
    }
}

bool GNSSProcessing::detectZeroVelocity(Vector3d &mean_acc, Vector3d &mean_gyr, Vector3d &cov_acc, Vector3d &cov_gyr){
    double N = 1.0;
    Vector3d cur_acc, cur_gyr;
    bool is_first_imu = true;
    for ( const auto &imu : imu_buffer_ ){
        auto &imu_acc = imu->linear_acceleration;
        auto &imu_gyr = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << imu_gyr.x, imu_gyr.y, imu_gyr.z;

        if(is_first_imu){
            mean_acc = cur_acc;
            mean_gyr = cur_gyr;
            is_first_imu = false;
        } 
        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc*(N-1.0) / N + (cur_acc-mean_acc).cwiseProduct(cur_acc-mean_acc)*(N-1.0)/(N*N);
        cov_gyr = cov_gyr*(N-1.0) / N + (cur_gyr-mean_gyr).cwiseProduct(cur_gyr-mean_gyr)*(N-1.0)/(N*N);
        N ++;
    }

    // std:fstream fout(DEBUG_FILE_DIR("detect.txt"), std::ios::app);
    // fout << setw(10) << imu_buffer_.front()->header.stamp.toSec() << " " << abs(mean_acc.norm()-G_m_s2) << " " << mean_gyr.norm() << endl << endl;
    // fout.close();
    
    // 使用比力和角速度的摸进行零速检测
    if((abs(mean_acc.norm()-G_m_s2) < ZERO_VELOCITY_ACC_THRESHOLD) && (abs(mean_gyr.norm()) < ZERO_VELOCITY_GYR_THRESHOLD))
        return true;
    // if((cov_acc.x() < ZERO_VELOCITY_ACC_THRESHOLD) && (cov_acc.y() < ZERO_VELOCITY_ACC_THRESHOLD) &&
    //     (cov_acc.z() < ZERO_VELOCITY_ACC_THRESHOLD) && (cov_gyr.x() < ZERO_VELOCITY_GYR_THRESHOLD) &&
    //     (cov_gyr.y() < ZERO_VELOCITY_GYR_THRESHOLD) && (cov_gyr.z() < ZERO_VELOCITY_GYR_THRESHOLD))
    //     return true;

    return false;
}

bool GNSSProcessing::Initialization(Vector3d &mean_acc, Vector3d &mean_gyr, Vector3d &cov_acc, Vector3d &cov_gyr, bool is_eular)
{
    if(imu_buffer_.size() < 40)
        return false;
    
    bool is_zero_velocity = detectZeroVelocity(mean_acc, mean_gyr, cov_acc, cov_gyr);
    if(is_zero_velocity){
        eular_[0] = -asin(mean_acc.y() / G_m_s2);
        eular_[1] = asin(mean_acc.x() / G_m_s2);

        is_has_zero_velocity = true;
        is_eular = false;

        imu_buffer_.clear();
        return true;
    }

    // if(last_gnss_.time==-1) return false;
    
    // if(!is_zero_velocity){
    //     Vector3d vel = gnss_.blh - last_gnss_.blh;
    //     cout << gnss_.blh << " " << last_gnss_.blh << " " << "vel: " << vel.norm() << endl;
    //     if(vel.norm() < MINMUM_ALIGN_VELOCITY)
    //         return false;
        
    //     if (!is_has_zero_velocity){
    //         eular_[0] = 0;
    //         eular_[1] = atan(-vel.z() / sqrt(vel.x() * vel.x() + vel.y() * vel.y()));
    //     }
    //     eular_[2] = atan2(vel.y(), vel.x());
    //     is_eular = true;
    // }

    cout << "Please keep static until imu init " << endl;
    return false;
}

void GNSSProcessing::optimize(){
    ceres::Problem prblem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;

    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;



}