#include "GNSS_Processing.h"
#include <fstream>

#define MINMUM_ALIGN_VELOCITY 0.5

GNSSProcessing::GNSSProcessing(ros::NodeHandle& nh)
{
    is_origin_set = false;
    is_has_yaw_ = false;
    new_gnss_ = false;
    // eular_ = Vector3d(0, 0, 0);
    eular_ = vector<double>(3, 0);
    anchor_ = Vector3d(0, 0, 0);

    string gnss_topic; vector<double> antlever;
    nh.param<string>("common/gnss_topic", gnss_topic, "navsat/fix");
    nh.param<vector<double>>("gnss/extrinsic", antlever, vector<double>());
    antlever_ << VEC_FROM_ARRAY(antlever);

    thread_opt_ = std::thread(&GNSSProcessing::optimize, this);

    gnss_sub_ = nh.subscribe(gnss_topic, 2000, &GNSSProcessing::input_gnss, this);
    odo_sub_ = nh.subscribe("/aft_mapped_to_init", 20000, &GNSSProcessing::input_path, this);
}

GNSSProcessing::~GNSSProcessing(){
    thread_opt_.detach();
}

void GNSSProcessing::input_gnss(const sensor_msgs::NavSatFix::ConstPtr& msg_in)
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
    
    gnss_mutex_.lock();
    gnss_queue_.push(gnss);
    gnss_mutex_.unlock();
}

void GNSSProcessing::input_path(const nav_msgs::Odometry::ConstPtr& msg_in){
    Vector3d pos = Vector3d(msg_in->pose.pose.position.x, msg_in->pose.pose.position.y, msg_in->pose.pose.position.z);
    double time = msg_in->header.stamp.toSec();

    // cout << "odo: " << std::fixed << std::setprecision(2) << time << endl;
    odo_mutex_.lock();
    gnss_mutex_.lock();
    odo_path_[time] = pos;

    while(!gnss_queue_.empty()){
        GNSS gnss_msg = gnss_queue_.front();
        double gnss_t = gnss_msg.time;
        if(gnss_t >= time-0.01 && gnss_t <= time+0.01){
            // ！待添加GNSS筛选条件
            // 与上一次gnss观测距离足够远，才认为是新的约束
            if(!gnss_buffer_.empty()){
                GNSS last_gnss = gnss_buffer_.end()->second;
                Vector3d diff = gnss_msg.blh - last_gnss.blh;

                if(diff.norm() < MINMUM_ALIGN_VELOCITY){
                    if(gnss_msg.std.norm() < last_gnss.std.norm()){
                        double rm_t = gnss_buffer_.end()->first;
                        gnss_buffer_.erase(rm_t);
                        gnss_buffer_[time] = gnss_msg;
                    }
                    gnss_queue_.pop();
                    break;
                }
            }

            gnss_buffer_[time] = gnss_msg;
            gnss_queue_.pop();

            std::ofstream outfile;
            outfile.open(DEBUG_FILE_DIR("gnss_size.txt"), std::ios::app);
            outfile << std::fixed << std::setprecision(6)<< gnss_buffer_.size() << endl;
            outfile.close();

            if(gnss_buffer_.size() > 1) new_gnss_ = true;
            break;
        }
        else if(gnss_t < time - 0.01)
            gnss_queue_.pop();
        else if(gnss_t > time + 0.01)
            break;
    }
    gnss_mutex_.unlock();
    odo_mutex_.unlock();
}

void GNSSProcessing::Initialization(Vector3d &mean_acc)
{
    Vector3d acc = mean_acc / mean_acc.norm();      // 归一化
    eular_[0] = -asin(acc.y());   // 绕x轴旋转 外旋x-y-z
    eular_[1] = asin(acc.x());    // 绕y轴旋转
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
    while(true){
        if(new_gnss_){
            new_gnss_ = false;

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.max_num_iterations = 50;

            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

            double yaw = eular_[2];
            problem.AddParameterBlock(&yaw, 1);    // 添加yaw角误差

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
                if(!is_has_yaw_) problem.SetParameterBlockConstant(t_array[i]);
            }

            if(is_has_yaw_){
                problem.SetParameterBlockConstant(&yaw);
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
                    ceres::CostFunction* gnss_function = new GnssFactor(iter_gnss->second, antlever_, eular_[0], eular_[1]);
                    auto id = problem.AddResidualBlock(gnss_function, loss_function, t_array[i], &yaw);
                    residual_block.push_back(std::make_pair(id, iter_gnss->second));
                    array_index.push_back(i);
                }
                i++;
            }

            ceres::Solve(options, &problem, &summary);
            
            if(!is_has_yaw_){
                cout << eular_[0] << " " << eular_[1] << endl;
                eular_[2] = yaw;
                is_has_yaw_ = true;
                cout << "yaw: " << yaw << endl;
            } else
            { // gnss卡方检测并给gnss观测赋予新的权重
                gnssOutlierCullingByChi2(problem, residual_block);

                i = 0;
                for(auto res : residual_block){
                    problem.RemoveResidualBlock(res.first);

                    ceres::CostFunction* gnss_function = new GnssFactor(res.second, antlever_, eular_[0], eular_[1]);
                    int index = array_index[i];
                    problem.AddResidualBlock(gnss_function, nullptr, t_array[index], &yaw);
                    i++;
                }

                ceres::Solve(options, &problem, &summary);
                // int index = t_array.size()-1;
                // cur_pos = Vector3d(t_array[index][0], t_array[index][1], t_array[index][2]);    // 方差怎么修改？
            
                // ceres::Covariance::Options options;
                // ceres::Covariance covariance(options);

                i=0;
                for(auto iter_odo=odo_path_.begin(); iter_odo != odo_path_.end(); iter_odo++, i++){
                    iter_odo->second = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                }

                if(gnss_buffer_.size() >= 10){
                    gnss_buffer_.clear();
                    i = 0; std::ofstream outfile;
                    outfile.open(DEBUG_FILE_DIR("gnss_odo.txt"), std::ios::app);

                    for(auto iter_odo = odo_path_.begin(); iter_odo != odo_path_.end(); iter_odo++, i++){
                        outfile << std::fixed << std::setprecision(6) << setw(20) << iter_odo->first << " " << iter_odo->second.transpose() << endl;
                    }
                    outfile.close();
                    odo_path_.clear();
                }
            }

            odo_mutex_.unlock();
            gnss_mutex_.unlock();
        }
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
    return true;
}
