#include "GNSS_Processing.h"
#include <fstream>

#define MINMUM_ALIGN_VELOCITY 0.5

GNSSProcessing::GNSSProcessing(ros::NodeHandle& nh)
{
    is_origin_set = false;
    is_has_yaw_ = false;
    new_gnss_ = false;
    ready_comp_ = false;
    // eular_ = Vector3d(0, 0, 0);
    yaw_ = 0.0;
    anchor_ = Vector3d(0, 0, 0);

    last_gnss_time_ = -1;

    string gnss_topic; vector<double> antlever;
    nh.param<string>("common/gnss_topic", gnss_topic, "navsat/fix");
    nh.param<vector<double>>("gnss/extrinsic", antlever, vector<double>());
    antlever_ << VEC_FROM_ARRAY(antlever);

    thread_opt_ = std::thread(&GNSSProcessing::optimize, this);

    gnss_sub_ = nh.subscribe(gnss_topic, 2000, &GNSSProcessing::input_gnss, this);
    // odo_sub_ = nh.subscribe("/aft_mapped_to_init", 20000, &GNSSProcessing::input_path, this);
}

GNSSProcessing::~GNSSProcessing(){
    thread_opt_.detach();
}

void GNSSProcessing::input_gnss(const sensor_msgs::NavSatFix::ConstPtr& msg_in)
{
    if(msg_in->status.status < 0) return;  // 不使用非固定解

    GNSS gnss;
    gnss.time = msg_in->header.stamp.toSec();
    gnss.blh  = Earth::blh2ecef(Vector3d(D2R * msg_in->latitude, D2R * msg_in->longitude, msg_in->altitude));
    // gnss.blh  = Earth::global2local(anchor_, D2R * Vector3d(msg_in->latitude, msg_in->longitude, msg_in->altitude));
    gnss.std  = Vector3d(msg_in->position_covariance[0], msg_in->position_covariance[4], msg_in->position_covariance[8]);

    // if(!is_origin_set){
    //     anchor_ = gnss.blh;
    //     is_origin_set = true;
    // }
    // auto enu = Earth::ecef2local(anchor_, gnss.blh);
    // ofstream fout(DEBUG_FILE_DIR("gnss_urban.txt"),ios::app);
    // fout << fixed << setprecision(3) << gnss.time << " " << enu.x() << " "
    //     << enu.y() << " " << enu.z() << " 0 0 0 0" << endl;
    // fout.close();

    gnss_mutex_.lock();
    gnss_queue_.push(gnss);
    gnss_mutex_.unlock();
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
            // std::cout << std::fixed << std::setprecision(3) << gnss.time << " ";
            gnss.blh  = ecef;
            // std::fstream fout(DEBUG_FILE_DIR("gnss.txt"),ios::app);
            // fout << std::fixed << std::setprecision(3) << gnss.time << " " << gnss.blh.x() << " " << gnss.blh.y() << " " << gnss.blh.z()
            //     << " 0 0 0 0" << std::endl;
            // fout.close();
            gnss.std  = enustd;
            gnss_mutex_.lock();
            gnss_queue_.push(gnss);
            gnss_mutex_.unlock();
        }
    }

    cout << "read gnss result successfully!" << endl;
}

void GNSSProcessing::input_path(const double &cur_time, const Eigen::Vector3d &position){
    Vector3d pos = position;
    double time = cur_time;

    odo_mutex_.lock();
    gnss_mutex_.lock();
    odo_path_[time] = pos;

    while(!gnss_queue_.empty()){
        GNSS &gnss_msg = gnss_queue_.front();
        double gnss_t = gnss_msg.time;
        if(!is_origin_set && gnss_t >= time-1 && gnss_t < time+1){
            anchor_ = gnss_msg.blh;
            is_origin_set = true;
        }

        if(gnss_t < time)
            gnss_queue_.pop();
        else if(gnss_t <= time+0.05){
            gnss_msg.blh = Earth::ecef2local(anchor_, gnss_msg.blh);
            new_gnss_ = true;
            break;
        }
        else break;
    }
    gnss_mutex_.unlock();
    odo_mutex_.unlock();
}

void GNSSProcessing::addIMUpos(const vector<Pose6D> &IMUpose, const double pcl_beg_time){
    if(new_gnss_){
        odo_mutex_.lock();
        gnss_mutex_.lock();
        GNSS gnss_msg = gnss_queue_.front();
        gnss_queue_.pop();
        double gnss_t = gnss_msg.time;

        // 如果两次gnss观测距离较近，就不使用本次gnss观测
        if(last_gnss_time_!=-1 && common::calc_dist(gnss_msg.blh, last_gnss_.blh) < 10){
            new_gnss_ = false;
            gnss_mutex_.unlock();
            odo_mutex_.unlock();
            return;
        }

        for(auto item : IMUpose){
            double time = pcl_beg_time + item.offset_time;

            if(gnss_t >= time-0.01 && gnss_t <= time+0.01){     // 阈值依据imu频率设定
                // 去除gnss飞点
                Eigen::Vector3d odo_pos(VEC_FROM_ARRAY(item.pos));
                // if(is_has_yaw_){      //is_has_yaw_
                //     Eigen::Vector3d gnss_pos;// = common::gnss_trans(gnss_msg.blh, yaw_);
                //     gnss_pos = gnss_msg.blh;
                    // if(common::calc_dist(odo_pos, gnss_pos)>3){
                    //     ofstream fout(DEBUG_FILE_DIR("delta.txt"),ios::app);
                    //     double dis = std::abs(odo_pos.norm() - gnss_pos.norm());
                    //     fout << fixed << setprecision(3) << odo_pos.transpose() << "; " << gnss_pos.transpose() << "; " << common::calc_dist(odo_pos, gnss_pos)
                    //         << "; " << dis << endl;
                    //     fout.close();
                    //     break;
                    // }
                // }

                odo_path_[time] = odo_pos;
                gnss_buffer_[time] = gnss_msg;
                last_gnss_ = gnss_msg;
                last_gnss_time_ = time;

                if(!is_has_yaw_ && gnss_buffer_.size() > 1){
                    Initialization();
                    is_has_yaw_ = true;
                    break;
                }
                
                std::unique_lock<std::mutex> lock(ready_mutex_);
                ready_comp_ = true;
                lock.unlock();
                ready_cv_.notify_all();     // 唤醒优化线程
                break;
            }
        }

        new_gnss_ = false;
        gnss_mutex_.unlock();
        odo_mutex_.unlock();
    }
}

void GNSSProcessing::Initialization()
{
    auto gnss_begin = gnss_buffer_.begin();
    auto gnss_end   = gnss_buffer_.end(); gnss_end--;
    Vector3d gnss_vel = gnss_end->second.blh - gnss_begin->second.blh;

    double begin_time = gnss_begin->first;
    double end_time   = gnss_end->first;
    auto odo_begin = odo_path_.find(begin_time);
    auto odo_end = odo_path_.find(end_time);
    Vector3d odo_vel = odo_end->second - odo_begin->second;

    // 计算yaw角大小及方向 gnss到odo
    Vector3d dir = gnss_vel.cross(odo_vel);
    double cos_yaw = gnss_vel.dot(odo_vel) / (gnss_vel.norm() * odo_vel.norm());
    yaw_ = acos(cos_yaw);
    yaw_ *= dir.y()>0? -1.0 :1.0;

    is_has_yaw_ = true;

    ofstream fout(DEBUG_FILE_DIR("init.txt"),ios::app);
    fout << fixed << setprecision(3) << "endtime: " << end_time << " gnss begin pos: " << gnss_begin->second.blh.transpose() << endl << "odo begin pos: " << odo_begin->second.transpose() << endl;
    
    fout << fixed << setprecision(3) << "gnss vel: " << gnss_vel.transpose() << endl << "odo vel: " << odo_vel.transpose() << endl;
    fout << fixed << setprecision(3) << begin_time << " yaw: " << yaw_*180/PI_M << endl;
    fout.close();
}

void GNSSProcessing::gnssOutlierCullingByChi2(ceres::Problem & problem,
                            vector<std::pair<ceres::ResidualBlockId, GNSS>> &residual_block){
    double chi2_threshold = 7.815;
    double cost, chi2;

    int outliers_counts = 0;
    for(auto &block : residual_block) {
        auto id     =   block.first;
        GNSS gnss  =   block.second;

        problem.EvaluateResidualBlock(id, false, &cost, nullptr, nullptr);
        chi2 = cost * 2;

        if(chi2 > chi2_threshold){
            double scale = sqrt(chi2 / chi2_threshold);
            gnss.std *= scale;
            outliers_counts++;
        }
    }

    if(outliers_counts){
        cout << "Detect " << outliers_counts << endl;
    }
}

bool GNSSProcessing::optimize(){
    while(true){        // 初始化的误差纳入到优化中了
        std::unique_lock<std::mutex> lock(ready_mutex_);
        ready_cv_.wait(lock, [&](){return ready_comp_;});
        ready_comp_ = false;
        lock.unlock();

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 50;

        ceres::Solver::Summary summary;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        odo_mutex_.lock();
        gnss_mutex_.lock();

        int len = odo_path_.size();
        double t_array[len][3];
        map<double, Vector3d>::iterator iter = odo_path_.begin();
        for(int i=0; i<len; i++, iter++){
            t_array[i][0] = iter->second.x();
            t_array[i][1] = iter->second.y();
            t_array[i][2] = iter->second.z();
            problem.AddParameterBlock(t_array[i], 3);
        }

        map<double, Vector3d>::iterator iter_odo, iter_odo_next;
        map<double, GNSS>::iterator iter_gnss;
        vector<std::pair<ceres::ResidualBlockId, GNSS>> residual_block;
        vector<int> array_index;
        int i = 0;
        // #par
        for (auto iter_odo = odo_path_.begin(); iter_odo != odo_path_.end(); iter_odo++, i++){
            iter_odo_next = iter_odo;
            iter_odo_next ++;
            if(i<len-1){
                Eigen::Vector3d tij = iter_odo_next->second - iter_odo->second;
                ceres::CostFunction* odo_function = new OdoFactor(tij, 0.1);
                problem.AddResidualBlock(odo_function, nullptr, t_array[i], t_array[i+1]);
            }

            double t = iter_odo->first;
            iter_gnss = gnss_buffer_.find(t);
            if (iter_gnss != gnss_buffer_.end()){
                ceres::CostFunction* gnss_function = new GnssFactor(iter_gnss->second, antlever_, yaw_);
                auto id = problem.AddResidualBlock(gnss_function, loss_function, t_array[i]);
                residual_block.push_back(std::make_pair(id, iter_gnss->second));
                array_index.push_back(i);
            }
        }

        ceres::Solve(options, &problem, &summary);

        if(1)
        { // gnss卡方检测并给gnss观测赋予新的权重
            gnssOutlierCullingByChi2(problem, residual_block);

            i = 0;
            for(auto res : residual_block){
                problem.RemoveResidualBlock(res.first);

                ceres::CostFunction* gnss_function = new GnssFactor(res.second, antlever_, yaw_);
                int index = array_index[i];
                problem.AddResidualBlock(gnss_function, nullptr, t_array[index]);
                i++;
            }

            ceres::Solve(options, &problem, &summary);
            // int index = t_array.size()-1;
            // cur_pos = Vector3d(t_array[index][0], t_array[index][1], t_array[index][2]);    // 方差怎么修改？
        
            // ceres::Covariance::Options options;
            // ceres::Covariance covariance(options);

            // i=0;
            // for(auto iter_odo=odo_path_.begin(); iter_odo != odo_path_.end(); iter_odo++, i++){
            //     iter_odo->second = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
            // }
            ofstream fout(DEBUG_FILE_DIR("delta.txt"),ios::app);
            fout << fixed << setprecision(3) << gnss_buffer_.size() << endl;
            fout.close();

            if(gnss_buffer_.size() >= 10){
                // gnss_buffer_.clear();
                i = 0; std::ofstream outfile;
                outfile.open(DEBUG_FILE_DIR("gnss_odo.txt"), std::ios::out);

                for(auto iter_odo = odo_path_.begin(); iter_odo != odo_path_.end(); iter_odo++, i++){
                    outfile << std::fixed << std::setprecision(6) << iter_odo->first << " " \
                        << t_array[i][0] << " " << t_array[i][1] << " " << t_array[i][2] \
                        << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << endl;
                }
                outfile.close();
                // odo_path_.clear();
            }
        }

        odo_mutex_.unlock();
        gnss_mutex_.unlock();
    }
    return true;
}
