#ifndef FACTORS_H
#define FACTORS_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include "common_lib.h"

class GnssFactor : public ceres::SizedCostFunction<3, 3>{
public:
    GnssFactor() = delete;

    explicit GnssFactor(const GNSS& gnss_data, Vector3d lever, double yaw)
     : gnss_(std::move(gnss_data)), lever_(std::move(lever)),
     yaw_(std::move(yaw)) {}

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Vector3d p{parameters[0][0], parameters[0][1], parameters[0][2]};

        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        // error = p + lever_ - common::gnss_trans(gnss_.blh, yaw_);
        error = p + lever_ - gnss_.blh;

        // ofstream fout(DEBUG_FILE_DIR("residual.txt"), std::ios::app);
        // fout << "gnss factor residual: " << error.transpose() << endl;
        // fout.close();

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
        }

        return true;
    }

private:
    GNSS gnss_;
    Vector3d lever_;
    double yaw_;            // gnss to odo
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

        // ofstream fout(DEBUG_FILE_DIR("residual.txt"), std::ios::app);
        // fout << "odo factor residual: " << error.transpose() << endl;
        // fout.close();

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