#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>

template <typename T> inline
void QuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[0] = q[0];
	q_inverse[1] = -q[1];
	q_inverse[2] = -q[2];
	q_inverse[3] = -q[3];
};


struct TError
{
	TError(double t_x, double t_y, double t_z, double q_w, double q_x, 
           double q_y ,double q_z, double i_x, double i_y, double i_z,
           double var_x, double var_y, double var_z)
				  :t_x(t_x), t_y(t_y), t_z(t_z), q_w(q_w), q_x(q_x), q_y(q_y),
                   q_z(q_z), i_x(i_x), i_y(i_y), i_z(i_z),
                   var_x(var_x), var_y(var_y), var_z(var_z){}

	template <typename T>
    // antlever表示t_imu2gnss 定义在imu坐标系下
    bool operator()(const T* antlever, const T* q_enu_global, T* residual) const{
        T t_enu_ant[3];
        t_enu_ant[0] = T(t_x);
        t_enu_ant[1] = T(t_y);
        t_enu_ant[2] = T(t_z);

        T q_global_enu[4];
        QuaternionInverse(q_enu_global, q_global_enu);

        T t_global_ant[3];
        ceres::QuaternionRotatePoint(q_global_enu, t_enu_ant, t_global_ant);

        T q_global_imu[4];
        q_global_imu[0] = T(q_w);
        q_global_imu[1] = T(q_x);
        q_global_imu[2] = T(q_y);
        q_global_imu[3] = T(q_z);

        T t_imu_ant[3];
        ceres::QuaternionRotatePoint(q_global_imu, antlever, t_imu_ant);

        residual[0] = (T(i_x) + t_imu_ant[0] - t_global_ant[0]) / T(var_x);
        residual[1] = (T(i_y) + t_imu_ant[1] - t_global_ant[1]) / T(var_y);
        residual[2] = (T(i_z) + t_imu_ant[2] - t_global_ant[2]) / T(var_z);

        return true;
    }
	// bool operator()(const T* tj, T* residuals) const
	// {
	// 	residuals[0] = (tj[0] - T(t_x)) / T(var_x);     // + T(level_x)
	// 	residuals[1] = (tj[1] - T(t_y)) / T(var_y);
	// 	residuals[2] = (tj[2] - T(t_z)) / T(var_z);

	// 	return true;
	// }

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                    const double q_w, const double q_x, const double q_y, const double q_z,
                                    const double i_x, const double i_y, const double i_z,
                                    const double var_x, const double var_y, const double var_z) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          TError, 3, 3, 4>(
	          	new TError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, i_x, i_y, i_z, var_x, var_y, var_z)));
	}

	double t_x, t_y, t_z, var_x, var_y, var_z; // gnss位置 enu坐标系
    double q_x, q_y, q_z, q_w;      // R_global2imu
    double i_x, i_y, i_z;           // imu位置 global坐标系
};

struct RelativeRTError
{
	RelativeRTError(double t_x, double t_y, double t_z, 
					double q_w, double q_x, double q_y, double q_z,
					double t_var, double q_var)
				  :t_x(t_x), t_y(t_y), t_z(t_z), 
				   q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
				   t_var(t_var), q_var(q_var){}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, const T* w_q_j, const T* tj, T* residuals) const
	{
		// 由参数模块求的帧间变化量
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

		T relative_q[4];	// 平移残差
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4];	// 旋转变化的误差
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q); 

		residuals[3] = T(2) * error_q[1] / T(q_var);	// 大部分四元数的误差表示为向量部分
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
									   const double q_w, const double q_x, const double q_y, const double q_z,
									   const double t_var, const double q_var) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          RelativeRTError, 6, 4, 3, 4, 3>(
	          	new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	double t_x, t_y, t_z, t_norm;
	double q_w, q_x, q_y, q_z;
	double t_var, q_var;

};