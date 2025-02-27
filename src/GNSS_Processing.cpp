#include "GNSS_Processing.h"

GNSSProcessing::GNSSProcessing(ros::NodeHandle& nh){
    is_init_ = false;
    is_anchor_ = false;
    new_gnss_ = false;
    anchor_ = Vector3d(0, 0, 0);
    rot_enu2global_ = Eigen::Matrix3d::Identity();

    last_gnss_time_ = -1;

    string gnss_topic; vector<double> antlever;
    nh.param<string>("common/gnss_topic", gnss_topic, "navsat/fix");
    nh.param<vector<double>>("gnss/extrinsic", antlever, vector<double>());
    antlever_ << VEC_FROM_ARRAY(antlever);
}

GNSSProcessing::~GNSSProcessing(){
}

void GNSSProcessing::readrtkresult(const string gnss_path){
    std::ifstream rtk_file(gnss_path);
    if(!rtk_file.is_open()){
        cout << "wrong path to rtk result!" << endl;
        return;
    }

    string line;
    while(std::getline(rtk_file, line)){

        if(line.find("END_HEAD")!=std::string::npos) break;
    }
    std::getline(rtk_file, line);

    std::getline(rtk_file, line);
    int col = 0; vector<int> col_idx;
    string value; stringstream ss(line);
    while(ss >> value){
        if(value == "Week") col_idx.push_back(col+1);
        else if(value == "GPSTime") col_idx.push_back(col+1);  // Data有两个值 所以要+1跳过
        else if(value == "X-ECEF") col_idx.push_back(col+1);
        else if(value == "Y-ECEF") col_idx.push_back(col+1);
        else if(value == "Z-ECEF") col_idx.push_back(col+1);
        else if(value == "SD-E") col_idx.push_back(col+5);    // ENU方向的方差 标准差要开方
        else if(value == "SD-N") col_idx.push_back(col+5);
        else if(value == "SD-U") col_idx.push_back(col+5);
        else if(value == "AR") col_idx.push_back(col+5);      // 固定情况，大于等于3时说明模糊度固定了
        col++;
    }
    std::getline(rtk_file, line);

    while(std::getline(rtk_file, line)){
        stringstream ss(line);
        col = 0;
        int week, AR;  double sow;
        Vector3d ecef; Vector3d enustd;
        
        while(ss >> value){
            if(col==col_idx[0]) week = atoi(value.c_str());
            else if(col==col_idx[1]) sow = stod(value);
            else if(col==col_idx[2]) ecef.x() = stod(value);
            else if(col==col_idx[3]) ecef.y() = stod(value);
            else if(col==col_idx[4]) ecef.z() = stod(value);
            else if(col==col_idx[5]) enustd.x() = stod(value);
            else if(col==col_idx[6]) enustd.y() = stod(value);
            else if(col==col_idx[7]) enustd.z() = stod(value);
            else if(col==col_idx[8]) AR = atoi(value.c_str());
            col++;
        }

        if(AR>=3){
            GNSS gnss;
            Earth::gps2unix(week, sow, gnss.time);
            gnss.blh  = ecef;
            gnss.std  = enustd;
            gnss_queue_.push(gnss);
        }
    }

    cout << "read gnss result successfully!" << endl;
}


void GNSSProcessing::addIMUpos(const vector<Pose6D> &IMUpose, const double pcl_beg_time){
    new_gnss_ = false;
    
    int len = IMUpose.size();
    double imu_begt = pcl_beg_time + IMUpose[0].offset_time;
    double imu_endt = pcl_beg_time + IMUpose[len-1].offset_time;

    while(!gnss_queue_.empty()){
        GNSS &gnss_msg = gnss_queue_.front();
        double gnss_t = gnss_msg.time;
        if(gnss_t < imu_begt)
            gnss_queue_.pop();
        else break;
    }

    GNSS &gnss_msg = gnss_queue_.front();
    double gnss_t = gnss_msg.time;
    if(gnss_t > imu_endt) return;

    if(!is_anchor_){
        anchor_ = gnss_msg.blh;
        is_anchor_ = true;
    }
    gnss_msg.blh = Earth::ecef2local(anchor_, gnss_msg.blh);    // enu坐标系

    // 如果两次gnss观测距离较近，就不使用本次gnss观测
    if(last_gnss_time_!=-1 && common::calc_dist(gnss_msg.blh, last_gnss_.blh) < 1)
        return;
    
    for(auto item : IMUpose){
        double imu_time = pcl_beg_time + item.offset_time;
        if(gnss_t >= imu_time-0.01 && gnss_t <= imu_time+0.01){

            Eigen::Vector3d odo_pos(VEC_FROM_ARRAY(item.pos));
            Eigen::Matrix3d odo_rot;
            odo_rot << MAT_FROM_ARRAY(item.rot);

            if(is_init_){
                gnss_ = gnss_msg;
                Eigen::Vector3d end_pos(VEC_FROM_ARRAY(IMUpose[len-1].pos));
                delta_pos_ = end_pos - odo_pos;
                new_gnss_ = true;
                // optimize();
            }
            else {
                odo_path_.emplace_back(odo_pos);
                odo_rot_.emplace_back(odo_rot);
                gnss_path_.emplace_back(gnss_msg);

                if(gnss_path_.size()>10){
                    Initialization();
                    is_init_ = true;
                    odo_path_.clear();
                    odo_rot_.clear();
                    gnss_path_.clear();
                    break;
                }
            }
            last_gnss_ = gnss_msg;
            last_gnss_time_ = imu_time;
        }
    }
}

void GNSSProcessing::Initialization(){
    // step1 估计yaw角
    int len = odo_path_.size();
    Vector3d gnss_vel = gnss_path_[len-1].blh - gnss_path_[0].blh;
    Vector3d odo_vel = odo_path_[len-1] - odo_path_[0];
    Vector3d dir = gnss_vel.cross(odo_vel);
    double cos_yaw = gnss_vel.dot(odo_vel) / (gnss_vel.norm() * odo_vel.norm());
    double yaw = dir.y()>0? -acos(cos_yaw) : acos(cos_yaw);
    rot_enu2global_ << cos_yaw, -sin(yaw), 0, sin(yaw), cos_yaw, 0, 0, 0, 1;
    auto q_rot = Eigen::Quaterniond(rot_enu2global_);

    // step1 ceres轨迹对齐
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

    double rot_array[4];
    double pos_array[3];
    pos_array[0] = antlever_(0);
    pos_array[1] = antlever_(1);
    pos_array[2] = antlever_(2);
    rot_array[0] = q_rot.w();
    rot_array[1] = q_rot.x();
    rot_array[2] = q_rot.y();
    rot_array[3] = q_rot.z();
    problem.AddParameterBlock(rot_array, 4, local_parameterization);
    problem.AddParameterBlock(pos_array, 3);

    for(int i=0; i<len; i++){
        Eigen::Quaterniond q_imu(odo_rot_[i]);
        ceres::CostFunction* gps_function = TError::Create(gnss_path_[i].blh(0), gnss_path_[i].blh(1), gnss_path_[i].blh(2),
                                                            q_imu.w(), q_imu.x(), q_imu.y(), q_imu.z(),
                                                            odo_path_[i](0), odo_path_[i](1), odo_path_[i](2),
                                                            gnss_path_[i].std(0), gnss_path_[i].std(1), gnss_path_[i].std(2));
        problem.AddResidualBlock(gps_function, loss_function, pos_array, rot_array);
    }
    ceres::Solve(options, &problem, &summary);

    // step2 计算外参
    rot_enu2global_ = Eigen::Quaterniond(rot_array[0], rot_array[1],
                    rot_array[2], rot_array[3]).toRotationMatrix();
    antlever_ = Eigen::Vector3d(pos_array[0], pos_array[1], pos_array[2]);

    std::fstream fout;
    fout.open(DEBUG_FILE_DIR("init_gnss.txt"), std::ios::app);
    fout << "gnss init sucess! " << std::endl;
    fout << std::fixed << std::setprecision(6) << antlever_.transpose() << std::endl << std::endl;
    fout << std::fixed << std::setprecision(6) << rot_enu2global_.transpose() << std::endl << std::endl;
    fout.close();
}

void GNSSProcessing::computeH(Eigen::MatrixXd &HTH, Eigen::MatrixXd &HTL, Eigen::Matrix3d &rot_end, Eigen::Vector3d pos_end){
    // step1 enu坐标系转到global坐标系
    V3D gnss_pos = gnss_.blh;
    gnss_pos = rot_enu2global_.transpose()*gnss_pos + delta_pos_; // 转到global系下
    
    // step2 计算雅各比矩阵
    Vector3d error; error.setZero();
    Matrix<double, 3, 6> Hsub; Hsub.setZero();
    Hsub.block<3,3>(0,3) = -1.0 * Matrix3d::Identity();
    Hsub.block<3,3>(0,0) = rot_end * SKEW_SYM_MATRX(antlever_);

    error = gnss_pos - pos_end + rot_end*antlever_;

    HTH.resize(6,6);    HTL.resize(6,1);
    HTH.setZero();      HTL.setZero();

    // step3 如果gnss位置和当前位置的误差大于阈值，则不使用此观测
    if(error.norm()>2.0)
        return;

    Matrix3d gnss_p; gnss_p.setIdentity();
    // gnss_p *= 10;
    gnss_p(0,0) = 1.0 / gnss_.std(0); // std的倒数作为对角阵
    gnss_p(1,1) = 1.0 / gnss_.std(1);
    gnss_p(2,2) = 100.0 / gnss_.std(2);

    HTH = Hsub.transpose() * gnss_p * Hsub;
    HTL = Hsub.transpose() * gnss_p * error;

    // std::fstream fout;
    // fout.open(DEBUG_FILE_DIR("gnss_optimize.txt"), std::ios::app);
    // fout << std::fixed << std::setprecision(5) << error.transpose() << std::endl << std::endl;
    // fout.close();
}

void GNSSProcessing::optimize(){
    // step1 enu坐标系转到global坐标系
    V3D gnss_pos = gnss_.blh;
    gnss_pos = rot_enu2global_.transpose()*gnss_pos + delta_pos_; // 转到global系下

    // step2 迭代优化
    Vector3d error; error.setZero();
    Matrix<double, 3, 6> Hsub; Hsub.setZero();
    Hsub.block<3,3>(0,3) = -1.0 * Matrix3d::Identity();
    Hsub.block<3,3>(0,0) = state->rot_end * SKEW_SYM_MATRX(antlever_);

    Matrix3d gnss_p; gnss_p.setIdentity();
    // gnss_p *= 10;
    gnss_p(0,0) = 1.0 / gnss_.std(0); // std的倒数作为对角阵
    gnss_p(1,1) = 1.0 / gnss_.std(1);
    gnss_p(2,2) = 1.0 / gnss_.std(2);
    
    Matrix<double, 6, 6> HTH = Hsub.transpose() * gnss_p * Hsub;
    Matrix<double, 6, 1> HPL;   HPL.setZero();

    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
    G.setZero();  H_T_H.setZero();  I_STATE.setIdentity();
    VD(DIM_STATE) solution;

    std::fstream fout;
    fout.open(DEBUG_FILE_DIR("gnss_optimize.txt"), std::ios::app);

    error = gnss_pos - state->pos_end + state->rot_end*antlever_;
    HPL = Hsub.transpose() * gnss_p * error;

    fout << std::fixed << std::setprecision(5) << error.transpose() << std::endl;
    
    H_T_H.block<6,6>(0,0) = HTH;
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov).inverse()).inverse();
    G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
    auto vec = *state_propagat - *state;
    solution = K_1.block<DIM_STATE,6>(0,0)*HPL+vec-G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);

    *state += solution;

    fout << std::fixed << std::setprecision(5) << gnss_.std.transpose() << std::endl << std::endl;
    fout.close();

    state->cov = (I_STATE - G) * state->cov;
    *state_propagat = *state;
}
