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
#include"IMU_Processing.h"

const bool time_list(PointType &x, PointType &y)
{
  return (x.curvature < y.curvature);
}

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_acc_scale = V3D(1, 1, 1);
  cov_gyr_scale = V3D(1, 1, 1);
  cov_bias_gyr  = V3D(0.1, 0.1, 0.1);     // 零偏不稳定性
  cov_bias_acc  = V3D(0.1, 0.1, 0.1);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU    = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::push_update_state(double offs_t, StatesGroup state)
{
  // V3D acc_tmp(last_acc), angvel_tmp(last_ang), vel_imu(state.vel_end), pos_imu(state.pos_end);
  // M3D R_imu(state.rot_end);
  // angvel_tmp -= state.bias_g;
  // acc_tmp   = acc_tmp * G_m_s2 / mean_acc.norm() - state.bias_a;
  // acc_tmp  = R_imu * acc_tmp + state.gravity;
  // IMUpose.push_back(set_pose6d(offs_t, acc_tmp, angvel_tmp, vel_imu, pos_imu, R_imu));
  V3D acc_tmp=acc_s_last, angvel_tmp=angvel_last, vel_imu(state.vel_end), pos_imu(state.pos_end);
  M3D R_imu(state.rot_end);
  IMUpose.push_back(set_pose6d(offs_t, acc_tmp, angvel_tmp, vel_imu, pos_imu, R_imu));
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU    = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov_scale(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

bool ImuProcess::detectZeroVelocity(const MeasureGroup &meas)
{
  int N = init_iter_num;
  // int N = 1;
  Vector3d cur_acc, cur_gyr;
  for (const auto &it : meas.imu){
    auto &imu_acc = it->linear_acceleration;
    auto &imu_gyr = it->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << imu_gyr.x, imu_gyr.y, imu_gyr.z;

    if(N == 1){ // 首帧
      mean_acc = cur_acc;
      mean_gyr = cur_gyr;
    }
    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc*(N-1.0) / N + (cur_acc-mean_acc).cwiseProduct(cur_acc-mean_acc)*(N-1.0)/(N*N);
    cov_gyr = cov_gyr*(N-1.0) / N + (cur_gyr-mean_gyr).cwiseProduct(cur_gyr-mean_gyr)*(N-1.0)/(N*N);
    N ++;
  }
  init_iter_num = N;

  // double acc_tmp = 0.0, w_tmp = 0.0;
  // for (const auto &it : meas.imu){
  //   auto &imu_acc = it->linear_acceleration;
  //   auto &imu_gyr = it->angular_velocity;
  //   cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
  //   cur_gyr << imu_gyr.x, imu_gyr.y, imu_gyr.z;

  //   acc_tmp += (cur_acc - mean_acc).transpose() * (cur_acc - mean_acc);
  //   w_tmp += cur_gyr.transpose() * cur_gyr;
  // }
  // double glrt = (acc_tmp) / N;

  // if(1){
  //   std::fstream fout(DEBUG_FILE_DIR("detectZero.txt"), std::ios::app);
  //   fout << N << endl;
  //   fout << std::fixed << std::setprecision(3) << "average value: " << abs(mean_acc.norm()-G_m_s2) << " " << abs(mean_gyr.norm()) << endl;
  //   fout << std::fixed << std::setprecision(3) << "cov value: " << cov_acc.transpose() << " " << cov_gyr.transpose() << endl;
  //   fout << std::fixed << std::setprecision(3) << "GLRT: " << glrt << endl;
  // }

  // 使用比力和角速度的摸进行零速检测 检测阈值与数据类型相关？如何自适应检测
  if((abs(mean_acc.norm()-G_m_s2) < 0.1) && (abs(mean_gyr.norm()) < 0.1))
    return true;
  // if((cov_acc.x() < ZERO_VELOCITY_ACC_THRESHOLD) && (cov_acc.y() < ZERO_VELOCITY_ACC_THRESHOLD) &&
  //     (cov_acc.z() < ZERO_VELOCITY_ACC_THRESHOLD) && (cov_gyr.x() < ZERO_VELOCITY_GYR_THRESHOLD) &&
  //     (cov_gyr.y() < ZERO_VELOCITY_GYR_THRESHOLD) && (cov_gyr.z() < ZERO_VELOCITY_GYR_THRESHOLD))
  //     return true;    
  
  return false;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  last_imu_   = meas.imu.back();

  bool is_zero_velocity = detectZeroVelocity(meas);

  if(!is_zero_velocity){
    cout << "Please wait until IMU init! ";
    Reset();
    return;
  }

  if(init_iter_num < 50) {
    printf("processing %.2f %\n", (double)init_iter_num/50.0 * 100);
    // std::cout << "processing %.1f" << init_iter_num/200 << endl;
    return;
  }

  if(is_zero_velocity)
  {
    // Vector3d gdir = Vector3d(0, 0, 1.0);  // imu朝向右前上
    // #ifdef MINI
    // gdir = Vector3d(-1.0, 0, 0);  // mini的imu朝向非常奇怪 下右后
    // #endif
    // state_inout.gravity = -1.0 * gdir * G_m_s2;  // UndistortPcl

    // // 首帧调整为水平方向，todo：杆臂也得调整
    // Vector3d diracc = mean_acc / mean_acc.norm();
    // Vector3d axis = gdir.cross(diracc);
    // axis  /= axis.norm();
    // double cosg = gdir.dot(diracc);
    // double ang = acos(cosg);
    // M3D R_g_imu = AngleAxisd(ang, axis).matrix();

    // state_inout.rot_end = R_g_imu;       // Rwi
    // state_inout.bias_g  = mean_gyr;

    state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;
    state_inout.bias_g  = mean_gyr;

    cov_acc = cov_acc * pow(G_m_s2 / mean_acc.norm(), 2);
    cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
    cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);

    imu_need_init_ = false;
  }
}

void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  // step1 存入数据并记录数据起始时间
  MeasureGroup meas;
  meas = lidar_meas.measures.back();

  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double pcl_beg_time = MAX(lidar_meas.lidar_beg_time, lidar_meas.last_update_time);
  start_timestamp_ = pcl_beg_time;

  /*** sort point clouds by offset time ***/
  // step2 判断从哪个雷达点开始处理，图像相关的雷达点没去畸变？？？
  pcl_out.clear();
  auto pcl_it = lidar_meas.lidar->points.begin() + lidar_meas.lidar_scan_index_now;
  auto pcl_it_end = lidar_meas.lidar->points.end(); 
  const double pcl_end_time = lidar_meas.is_lidar_end? 
                                        lidar_meas.lidar_beg_time + lidar_meas.lidar->points.back().curvature / double(1000):
                                        lidar_meas.lidar_beg_time + lidar_meas.measures.back().img_offset_time;
  const double pcl_offset_time = lidar_meas.is_lidar_end? 
                                        (pcl_end_time - lidar_meas.lidar_beg_time) * double(1000):
                                        0.0;
  while (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time)
  {
    pcl_out.push_back(*pcl_it);
    pcl_it++;
    lidar_meas.lidar_scan_index_now++;
  }
  lidar_meas.last_update_time = pcl_end_time;
  if (lidar_meas.is_lidar_end)
  {
    lidar_meas.lidar_scan_index_now = 0;
  }
  /*** Initialize IMU pose ***/
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  
  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu != v_imu.end()-1 ; it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    // 零偏修正
    angvel_avr -= state_inout.bias_g;
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;    // G_m_s2 / mean_acc.norm()相当于修正了一个线性比例因子误差

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    /* covariance propagation */
    M3D acc_avr_skew;
    M3D Exp_f   = Exp(angvel_avr, dt);
    acc_avr_skew<<SKEW_SYM_MATRX(acc_avr);

    F_x.setIdentity();
    cov_w.setZero();

    F_x.block<3,3>(0,0)  = Exp(angvel_avr, - dt);
    F_x.block<3,3>(0,9)  = - Eye3d * dt;
    F_x.block<3,3>(3,6)  = Eye3d * dt;
    F_x.block<3,3>(6,0)  = - R_imu * acc_avr_skew * dt;
    F_x.block<3,3>(6,12) = - R_imu * dt;
    F_x.block<3,3>(6,15) = Eye3d * dt;

    cov_w.block<3,3>(0,0).diagonal()   = cov_gyr * dt * dt;
    cov_w.block<3,3>(6,6)              = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3,3>(9,9).diagonal()   = cov_bias_gyr * dt * dt; // bias gyro covariance
    cov_w.block<3,3>(12,12).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU  先将加速度转到global坐标系下，然后修正重力*/
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  if (imu_end_time>pcl_beg_time)
  {
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
  }
  else
  {
    double note = pcl_end_time > pcl_beg_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - pcl_beg_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
  }

  last_imu_ = v_imu.back();
  last_lidar_end_time_ = pcl_end_time;

  M3D extR_Ri(Lid_rot_to_IMU.transpose() * state_inout.rot_end.transpose());
  V3D exrR_extT(Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
  
  if (pcl_out.points.size() < 1) return;
  /*** undistort each lidar point (backward propagation) ***/
  // 雷达点云去畸变
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    acc_imu<<VEC_FROM_ARRAY(head->acc);
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(head->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos_end);

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D P_compensate = (extR_Ri * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - exrR_extT);

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  MeasureGroup meas = lidar_meas.measures.back();   // 最新一帧imu观测
  // imu初始化
  if (imu_need_init_)
  {
    if(meas.imu.empty()) {return;};
    /// imu初始化
    IMU_init(meas, stat, init_iter_num);    // init_iter_num用作初始化的imu帧数

    return;
  }
  // 点云去畸变
  UndistortPcl(lidar_meas, stat, *cur_pcl_un_);
}