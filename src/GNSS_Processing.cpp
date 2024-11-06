#include "GNSS_Processing.h"
#include <fstream>

#define MINMUM_ALIGN_VELOCITY 0.5

GNSSProcessing::GNSSProcessing(ros::NodeHandle& nh)
{
    is_origin_set = false;
    is_has_yaw_ = false;
    new_gnss_ = false;
    // eular_ = Vector3d(0, 0, 0);
    yaw_ = 0.0;
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
        anchor_ = D2R * anchor_;
        // G_m_s2  = earth::gravity(anchor_);
        is_origin_set = true;
    }

    GNSS gnss;
    gnss.time = msg_in->header.stamp.toSec();
    gnss.blh  = Earth::global2local(anchor_, D2R * Vector3d(msg_in->latitude, msg_in->longitude, msg_in->altitude));
    gnss.std  = Vector3d(msg_in->position_covariance[0], msg_in->position_covariance[4], msg_in->position_covariance[8]);
    
    // ofstream fout(DEBUG_FILE_DIR("value.txt"),ios::app);
    // fout << fixed << setprecision(3) << "time: " << gnss.time << " gnss pos: " << gnss.blh.transpose() << endl;
    // fout.close();

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

    // ofstream fout(DEBUG_FILE_DIR("value.txt"),ios::app);
    // fout << fixed << setprecision(3) << "time: " << time << " odo pos: " << pos.transpose() << endl;
    // fout.close();

    while(!gnss_queue_.empty()){
        GNSS gnss_msg = gnss_queue_.front();
        double gnss_t = gnss_msg.time;
        if(gnss_t >= time-0.01 && gnss_t <= time+0.01){
            // ！待添加GNSS筛选条件
            // 与上一次gnss观测距离足够远，才认为是新的约束
            // if(!gnss_buffer_.empty()){
            //     auto last_gnss_it = gnss_buffer_.find(last_gnss_time_);
            //     GNSS last_gnss = last_gnss_it->second;
            //     Vector3d diff = gnss_msg.blh - last_gnss.blh;

            //     if(diff.norm() < MINMUM_ALIGN_VELOCITY){
            //         if(gnss_msg.std.norm() < last_gnss.std.norm()){
            //             gnss_buffer_.erase(last_gnss_time_);
            //             gnss_buffer_[time] = gnss_msg;
            //             last_gnss_time_ =  time;
            //         }
            //         gnss_queue_.pop();
            //         break;
            //     }
            // }

            gnss_buffer_[time] = gnss_msg;
            gnss_queue_.pop();
            last_gnss_time_ =   time;

            // ofstream fout(DEBUG_FILE_DIR("value.txt"),ios::app);
            // fout << fixed << setprecision(3) << "time: " << time << " gnss pos: " << gnss_msg.blh.transpose() << " odo pos: " << pos.transpose() << endl;
            // fout.close();

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
    yaw_ *= dir.y()>0? 1.0 :-1.0;

    is_has_yaw_ = true;

    ofstream fout(DEBUG_FILE_DIR("init.txt"),ios::app);
    fout << fixed << setprecision(3) << "endtime: " << end_time << " gnss begin pos: " << gnss_begin->second.blh.transpose() << endl << "odo begin pos: " << odo_begin->second.transpose() << endl;
    
    fout << fixed << setprecision(3) << "gnss vel: " << gnss_vel.transpose() << endl << "odo vel: " << odo_vel.transpose() << endl;
    fout << fixed << setprecision(3) << begin_time << " yaw: " << yaw_ << endl;
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
    while(true){
        if(new_gnss_){
            new_gnss_ = false;

            if(!is_has_yaw_){
                Initialization();
                continue;
            }

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
                i++;
            }

            ceres::Solve(options, &problem, &summary);
            
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
