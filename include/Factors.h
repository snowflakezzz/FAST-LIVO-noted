#ifndef FACTORS_H
#define FACTORS_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

class GnssFactor : public ceres::SizedCostFunction<3, 3, 1>{
public:
    GnssFactor() = delete;

    explicit GnssFactor(const GNSS& gnss_data, Vector3d lever, double pitch, double roll)
     : gnss_(std::move(gnss_data)), lever_(std::move(lever)),
     pitch_(std::move(pitch)), roll_(std::move(roll)) {}

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Vector3d p{parameters[0][0], parameters[0][1], parameters[0][2]};
        double yaw = parameters[1][0];

        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        // Vector3d euler_angles = Vector3d(roll_, pitch_, yaw);
        Eigen::Matrix3d R_z, R_y, R_x;
        R_x << 1, 0, 0, 0, cos(roll_), -sin(roll_), 0, sin(roll_), cos(roll_);
        R_y << cos(pitch_), 0, sin(pitch_), 0, 1, 0, -sin(pitch_), 0, cos(pitch_);
        R_z << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;

        Eigen::Matrix3d R_gnss_global = R_z*R_y*R_x;

        error = p + lever_ - R_gnss_global * gnss_.blh;

        Matrix3d sqrt_info_ = Matrix3d::Identity();     // 设定权重
        sqrt_info_(0,0) /= gnss_.std[0];
        sqrt_info_(1,1) /= gnss_.std[1];
        sqrt_info_(2,2) /= gnss_.std[2];

        error = sqrt_info_ * error;

        if(jacobians) {
            if(jacobians[0]) {
                jacobians[0][0] = 1.0;  jacobians[0][1] = 0.0;  jacobians[0][2] = 0.0;
                jacobians[0][3] = 0.0;  jacobians[0][4] = 1.0;  jacobians[0][5] = 0.0;
                jacobians[0][6] = 0.0;  jacobians[0][7] = 0.0;  jacobians[0][8] = 1.0;
            }
            if(jacobians[1]) {
                Eigen::Matrix3d J_yaw;
                J_yaw << -sin(yaw), -cos(yaw), 0, cos(yaw), -sin(yaw), 0, 0, 0, 0;
                auto tmp = -1.0 * J_yaw * R_y * R_x * gnss_.blh;
                jacobians[1][0] = tmp.x();
                jacobians[1][1] = tmp.y();
                jacobians[1][2] = tmp.z();
            }
        }

        return true;
    }

private:
    GNSS gnss_;
    Vector3d lever_;
    double pitch_;
    double roll_;
};

class OdoFactor : public ceres::SizedCostFunction<3, 3, 3>{
public:
    OdoFactor() = delete;

    explicit OdoFactor(Vector3d t, double t_var)
     : delta_t_(std::move(t)), t_var_(std::move(t_var)) {}

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override{
        Vector3d p1{parameters[0][0], parameters[0][1], parameters[0][2]};
        Vector3d p2{parameters[1][0], parameters[1][1], parameters[1][2]};
        
        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        error = p2 - p1 - delta_t_;
        error = error*t_var_;

        if(jacobians) {
            if(jacobians[0]) {
                jacobians[0][0] = -1.0;  jacobians[0][1] = 0.0;  jacobians[0][2] = 0.0;
                jacobians[0][3] = 0.0;  jacobians[0][4] = -1.0;  jacobians[0][5] = 0.0;
                jacobians[0][6] = 0.0;  jacobians[0][7] = 0.0;  jacobians[0][8] = -1.0;
            }
            if(jacobians[1]) {
                jacobians[1][0] = 1.0;  jacobians[1][1] = 0.0;  jacobians[1][2] = 0.0;
                jacobians[1][3] = 0.0;  jacobians[1][4] = 1.0;  jacobians[1][5] = 0.0;
                jacobians[1][6] = 0.0;  jacobians[1][7] = 0.0;  jacobians[1][8] = 1.0;
            }
        }
        return true;
    }

private:
    Vector3d delta_t_;// 帧间变化量
    double t_var_;    // 权重
};

#endif