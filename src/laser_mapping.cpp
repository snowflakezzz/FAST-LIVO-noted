#include "laser_mapping.h"
#include <vikit/camera_loader.h>

LaserMapping::LaserMapping()
{
    pcl_wait_pub = boost::make_shared<PointCloudXYZI>();
    Lidar_offset_to_IMU = Vector3d(0, 0, 0);
    position_last = Vector3d(0, 0, 0);
    Lidar_rot_to_IMU = Matrix3d::Identity();

    p_pre = std::make_shared<Preprocess>();
    p_imu = std::make_shared<ImuProcess>();

    sub_map_cur_frame_point = boost::make_shared<PointCloudXYZI>();
    feats_undistort = boost::make_shared<PointCloudXYZI>();
    feats_down_body = boost::make_shared<PointCloudXYZI>();
    feats_down_world = boost::make_shared<PointCloudXYZI>();

    pcl_wait_save = boost::make_shared<PointCloudXYZRGB>();
    pcl_wait_save_lidar = boost::make_shared<PointCloudXYZI>();
    laserCloudOri = boost::make_shared<PointCloudXYZI>();

    flg_first_scan = true;

    pathout.open(DEBUG_FILE_DIR("tum.txt"), std::ios::out);
}

LaserMapping::~LaserMapping(){ 
    pathout.close();
    if(loop_en){
        thread_loop_.detach();//join();
    }
};

void LaserMapping::Run(){
    // step1 数据对齐，存入LidarMeasures
    if (!sync_packages(LidarMeasures)){
        cv::waitKey(1);
        return;
    }

    if (flg_reset)
    {
        ROS_WARN("reset when rosbag play back");
        p_imu->Reset();
        flg_reset = false;
        return;
    }

    // 统计各模块运行时间
    // match_time = kdtree_search_time = kdtree_search_counter = solve_time = solve_const_H_time = svd_time   = 0;
    double t0,t1,t2,t3,t4,t5; //,match_start, solve_start, svd_time;
    t0 = omp_get_wtime();

    // step2 imu初始化，点云去畸变到雷达扫描结束或图像帧处
    p_imu->Process2(LidarMeasures, state, feats_undistort); 
    state_propagat = state;

    if (lidar_selector->debug) LidarMeasures.debug_show();

    // cout << feats_undistort->size() << endl;
    // imu还未初始化成功
    if (feats_undistort->empty() || (feats_undistort == nullptr))
    {
        if (!fast_lio_is_ready)
        {
            first_lidar_time = LidarMeasures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            LidarMeasures.measures.clear();
            cout<<"FAST-LIO not ready"<<endl;
            cout << "wait for imu init!" << endl;
            return;
        }
    }

    fast_lio_is_ready = true;
    flg_EKF_inited = (LidarMeasures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                    false : true;

    if(bgnss_en)
        p_gnss->addIMUpos(p_imu->IMUpose, p_imu->start_timestamp_);
    
    // step3 处理视觉观测信息，在lidar_selector中进行视觉追踪、位姿优化
    if (!LidarMeasures.is_lidar_end) 
    {
        cout<<"[ VIO ]: Raw feature num: "<<pcl_wait_pub->points.size() << "." << endl;
        // if (first_lidar_time < 10) return; // 累积足够多的雷达点
        if (img_en) {
            euler_cur = RotMtoEuler(state.rot_end);
            // step4.1 视觉追踪、残差构建、state位姿优化更新
            lidar_selector->detect(LidarMeasures.measures.back().img, pcl_wait_pub);
            int size_sub = lidar_selector->sub_map_cur_frame_.size();

            // step4.2 点云可视化
            sub_map_cur_frame_point->clear();
            for(int i=0; i<size_sub; i++)
            {
                PointType temp_map;
                temp_map.x = lidar_selector->sub_map_cur_frame_[i]->pos_[0];
                temp_map.y = lidar_selector->sub_map_cur_frame_[i]->pos_[1];
                temp_map.z = lidar_selector->sub_map_cur_frame_[i]->pos_[2];
                temp_map.intensity = 0.;
                sub_map_cur_frame_point->push_back(temp_map);
            }
            cv::Mat img_rgb = lidar_selector->img_cp;       // 绘制了点的图像
            cv_bridge::CvImage out_msg;
            out_msg.header.stamp = ros::Time().fromSec(LidarMeasures.last_update_time);//ros::Time::now();
            out_msg.encoding = sensor_msgs::image_encodings::BGR8;
            out_msg.image = img_rgb;
            img_pub.publish(out_msg.toImageMsg());

            if(loop_en) this->save_keyframe_factor();

            if(img_en) publish_frame_world_rgb();
            publish_visual_world_sub_map();
            
            geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
            publish_odometry();
            euler_cur = RotMtoEuler(state.rot_end);

            if(bgnss_en) p_gnss->input_path(LidarMeasures.last_update_time, state.pos_end);
        }
        return;
    }

    // step4 点云降采样，并初始化局部地图
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    // feats_down_body = feats_undistort;

    #ifdef USE_ikdtree
    if(ikdtree.Root_Node == nullptr)
    {
        if(feats_down_body->points.size() > 5)
        {
            ikdtree.set_downsample_param(filter_size_map_min);
            ikdtree.Build(feats_down_body->points);
        }
        return;
    }
    int featsFromMapNum = ikdtree.size();
    #else
    if (flg_first_scan){
        if(feats_down_body->points.size() > 5) {
            ivox_->AddPoints(feats_down_body->points);
            flg_first_scan = false;
        }
        return;
    }
    #endif

    feats_down_size = feats_down_body->points.size();
    cout<<"[ LIO ]: Raw feature num: "<<feats_undistort->points.size()<<" downsamp num "<<feats_down_size<< endl;//<<" Map num: "<<featsFromMapNum<< "." << endl;

    feats_down_world->resize(feats_down_size);

    t1 = omp_get_wtime();
    if (lidar_en)
    {
        euler_cur = RotMtoEuler(state.rot_end);
    }

    Nearest_Points.resize(feats_down_size);
    int  rematch_num = 0;
    nearest_search_en = true;

    t2 = omp_get_wtime();

    V3D rot_add, t_add;                                 // 误差状态量
    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
    G.setZero();  H_T_H.setZero();  I_STATE.setIdentity();
    VD(DIM_STATE) solution;
    bool EKF_stop_flg = 0;
    // step5 基于lidar观测的迭代误差卡尔曼滤波更新
    if(lidar_en){
        for (int iterCount = -1; iterCount < NUM_MAX_ITERATIONS && flg_EKF_inited; iterCount++ )
        {
            // step5.1 构建残差及雅各比
            MatrixXd HTH;  VectorXd HTL;
            h_share_model(HTH, HTL);

            MatrixXd K(DIM_STATE, effct_feat_num);
            EKF_stop_flg = false;
            flg_EKF_converged = false;

            // step5.2 误差状态卡尔曼滤波更新
            auto &&HTz = HTL;
            H_T_H.block<6,6>(0,0) = HTH;
            MD(DIM_STATE, DIM_STATE) &&K_1 = \
                    (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            auto vec = state_propagat - state;
            solution = K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);
        
            int minRow, minCol;
            if(0)//if(V.minCoeff(&minRow, &minCol) < 1.0f)
            {
                VD(6) V = H_T_H.block<6,6>(0,0).eigenvalues().real();
                cout<<"!!!!!! Degeneration Happend, eigen values: "<<V.transpose()<<endl;
                EKF_stop_flg = true;
                solution.block<6,1>(9,0).setZero();
            }
            
            state += solution;

            rot_add = solution.block<3,1>(0,0);
            t_add   = solution.block<3,1>(3,0);
            // 改正量小于阈值，终止迭代
            if ((rot_add.norm() * R2D < 0.01) && (t_add.norm() * 100 < 0.015))
            {
                flg_EKF_converged = true;
            }

            euler_cur = RotMtoEuler(state.rot_end);

            // step5.3 判断是否要重新进行最近临点搜索及残差构建
            nearest_search_en = false;
            if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
            {
                nearest_search_en = true;
                rematch_num ++;
            }

            if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1)))
            {
                if (flg_EKF_inited)
                {
                    /*** Covariance Update ***/
                    state.cov = (I_STATE - G) * state.cov;
                    total_distance += (state.pos_end - position_last).norm();   // 系统运行距离
                    position_last = state.pos_end;
                    geoQuat = tf::createQuaternionMsgFromRollPitchYaw           // 更新后的姿态
                                (euler_cur(0), euler_cur(1), euler_cur(2));

                    VD(DIM_STATE) K_sum  = K.rowwise().sum();
                    VD(DIM_STATE) P_diag = state.cov.diagonal();
                }
                EKF_stop_flg = true;
            }
            if (EKF_stop_flg) break;
        }
    }

    euler_cur = RotMtoEuler(state.rot_end);
    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
    publish_odometry();
    
    if(bgnss_en) p_gnss->input_path(LidarMeasures.last_update_time, state.pos_end);

    t3 = omp_get_wtime();
    map_incremental();
    t5 = omp_get_wtime();
    kdtree_incremental_time = t5 - t3; // + readd_time;
    ofstream out_file(DEBUG_FILE_DIR("kdtree_incremental_time.txt"), std::ios::app);
    out_file << kdtree_incremental_time << endl;
    out_file.close();

    PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);          
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }
    std::unique_lock<std::mutex> lock(m_loop_rady);
    *pcl_wait_pub = *laserCloudWorld;
    lock.unlock();

    if(!img_en) publish_frame_world();
    publish_effect_world();
    publish_path();

    // frame_num ++;
    // aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;

    T1[time_log_counter] = LidarMeasures.lidar_beg_time;
    s_plot[time_log_counter] = 0.0; //aver_time_consu;
    s_plot2[time_log_counter] = kdtree_incremental_time;
    s_plot3[time_log_counter] = 0.0; // kdtree_search_time/kdtree_search_counter;
    s_plot4[time_log_counter] = 0.0; //featsFromMapNum;
    s_plot5[time_log_counter] = t5 - t0;
    time_log_counter ++;
}

void LaserMapping::Finish(){
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("rgb_scan_all.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current rgb scan saved" << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (pcl_wait_save_lidar->size() > 0 && pcd_save_en)
    {
        string file_name = string("intensity_sacn_all.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current intensity scan saved" << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_lidar);
    }

    #ifndef DEPLOY
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
    FILE *fp2;
    string log_dir = DEBUG_FILE_DIR("class_fast_livo_time_log.csv");
    fp2 = fopen(log_dir.c_str(),"w");
    fprintf(fp2,"time_stamp, average time, incremental time, search time,fov check time, total time, alpha_bal, alpha_del\n");
    for (int i = 0;i<time_log_counter; i++){
        fprintf(fp2,"%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%f,%f\n",T1[i],s_plot[i],s_plot2[i],s_plot3[i],s_plot4[i],s_plot5[i],s_plot6[i],s_plot7[i]);
        t.push_back(T1[i]);
        s_vec.push_back(s_plot[i]);
        s_vec2.push_back(s_plot2[i]);
        s_vec3.push_back(s_plot3[i]);
        s_vec4.push_back(s_plot4[i]);
        s_vec5.push_back(s_plot5[i]);
        s_vec6.push_back(s_plot6[i]);
        s_vec7.push_back(s_plot7[i]);
    }
    fclose(fp2);
    #endif
}

void LaserMapping::map_incremental()
{
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = feats_down_body->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        /* decide if need add to map */
        PointType &point_world = feats_down_world->points[i];
        if (!Nearest_Points[i].empty() && flg_EKF_inited) {
            const PointVector &points_near = Nearest_Points[i];
            
            // 计算该点对应voxel的中心点
            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }   // 该点最近邻点离中心点太远，直接添加

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    #ifdef USE_ikdtree
    ikdtree.Add_Points(feats_down_world->points, true);
    #else
    ivox_->AddPoints(points_to_add);
    ivox_->AddPoints(point_no_need_downsample);
    #endif
}

#ifdef USE_VGICP
void LaserMapping::caculate_covariance(PointCloudXYZI::Ptr &cloud_in, vector<M3D> &covariances){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::copyPointCloud(*cloud_in, *cloud);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    covariances.resize(cloud->size());

    for (int i = 0; i < cloud->size(); i++) {
        std::vector<int> k_indices;         // 存储最近邻点
        std::vector<float> k_sq_distances;  // 最近邻点到查询点的距离平方
        kdtree.nearestKSearch(cloud->at(i), 15, k_indices, k_sq_distances);

        Eigen::Matrix<double, 4, -1> neighbors(4, 15);
        for (int j = 0; j < k_indices.size(); j++) {
            neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().cast<double>();
        }

        neighbors.colwise() -= neighbors.rowwise().mean().eval(); // .rowwise().mean()计算每行均值.eval()将计算结果转换为与 neighbors 矩阵相同大小的矩阵
        Eigen::Matrix4d cov = neighbors * neighbors.transpose() / 15; // 4*4
        covariances[i] = cov.block<3, 3>(0, 0);
    }
}

// VGICP构建观测方程
void LaserMapping::h_share_model(MatrixXd &HPH, VectorXd &HPL)
{
    vector<M3D> input_cov;
    caculate_covariance(feats_down_body, input_cov);

    HPH.resize(6,6); HPH.setZero();
    HPL.resize(6); HPL.setZero();

    int k_corre = 15;    // 用于计算方差的最近邻点个数
    for(int i=0; i<feats_down_size; i++){
        // step1 计算点云的世界坐标
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];
        V3D p_body(point_body.x, point_body.y, point_body.z);
        pointBodyToWorld(&point_body, &point_world);
        V3D p_word(point_world.x, point_world.y, point_world.z);

        // step2 最近邻搜索
        auto &points_near = Nearest_Points[i];
        uint8_t search_flag = 0;  
        double search_start = omp_get_wtime();
        if (nearest_search_en){
            ivox_->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);     // ivoxel搜索的时候已经考虑到最远距离
            kdtree_search_counter ++;                        
        }
        if (points_near.size() < NUM_MATCH_POINTS) continue;
        int points_num = points_near.size();

        // step3 计算最近邻点的方差
        M3D cov_B = M3D::Zero();   V3D mean_B = V3D::Zero();
        mean_B = V3D(points_near[0].x, points_near[0].y, points_near[0].z);
        PointVector point_near_near(k_corre);
        ivox_->GetClosestPoint(points_near[0], point_near_near, k_corre);

        int neighbor_num = point_near_near.size();
        Eigen::Matrix<double, 3, -1> neighbors(3, neighbor_num);
        for(int j=0; j<neighbor_num; ++j){
            neighbors.col(j) = V3D(point_near_near[j].x, point_near_near[j].y, point_near_near[j].z);
        }
        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        cov_B = neighbors * neighbors.transpose() / neighbor_num;

        // step4 计算方差
        M3D cov_A = input_cov[i];

        // step5 残差计算
        M3D rotation = state.rot_end * Lidar_rot_to_IMU;
        M3D RCR = cov_B + rotation*cov_A*rotation.transpose();
        M3D mahalanobis = RCR.inverse();

        V3D error = mean_B - p_word;
        Eigen::Matrix<double, 3, 6> dedx = Eigen::Matrix<double, 3, 6>::Zero();
        dedx.block<3, 3>(0, 0) = common::skewd(p_word);
        dedx.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

        double w = std::sqrt(points_num);
        // double w = 1.0;
        Matrix<double, 6, 6> Hi = w * dedx.transpose() * mahalanobis * dedx;
        Matrix<double, 6, 1> bi = w * dedx.transpose() * mahalanobis * error;

        HPH += Hi;
        HPL += bi;
    }
}

#else
// 返回观测矩阵构建成功
void LaserMapping::h_share_model(MatrixXd &HPH, VectorXd &HPL)
{
    PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
    normvec->resize(feats_down_size);
    PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());        // 筛选后正确的约束平面
    vector<bool> point_selected_surf(feats_down_size, true);        // 记录某个拟合出的观测面是否被使用

    double res_mean_last = 0.05;
    double total_residual = 0.0;
    PointCloudXYZI ().swap(*laserCloudOri);
    vector<double> res_last(feats_down_size, 1000.0);

    // PointCloudXYZI ().swap(*corr_normvect);
    
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for(int i = 0; i < feats_down_size; i++){
        // step1 计算点云的世界坐标
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];
        V3D p_body(point_body.x, point_body.y, point_body.z);
        pointBodyToWorld(&point_body, &point_world);

        // step2 最近邻搜索
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);  // 存储每个点距离的平方
        auto &points_near = Nearest_Points[i];
        uint8_t search_flag = 0;  
        double search_start = omp_get_wtime();
        if (nearest_search_en){
            #ifdef USE_ikdtree
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
            #else
            ivox_->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);     // ivoxel搜索的时候已经考虑到最远距离
            #endif
            kdtree_search_counter ++;                        
        }

        if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS) continue;

        // step3 平面拟合及残差构建，拟合的平面参数存储在normvec中
        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    // step4 筛选有效约束
    effct_feat_num = 0;
    laserCloudOri->resize(feats_down_size);
    corr_normvect->reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i] && (res_last[i] <= 2.0))
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }
    std::cout << "lidar effrct feat num: " << effct_feat_num << std::endl;
    res_mean_last = total_residual / effct_feat_num;

    // step5 计算观测雅各比
    HPH.resize(6, 6); HPH.setZero();
    HPL.resize(6); HPL.setZero();

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this(laser_p.x, laser_p.y, laser_p.z);
        point_this = Lidar_rot_to_IMU*point_this + Lidar_offset_to_IMU;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);

        Eigen::Matrix<double, 1, 6> Hsub;
        Hsub << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;
        HPH += Hsub.transpose() * Hsub;

        /*** Measuremnt: distance to the closest surface/corner ***/
        double error = - norm_p.intensity;
        HPL += Hsub.transpose() * error;
    }
}
#endif

void LaserMapping::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);        // 雷达坐标系
    V3D p_global(state.rot_end * (Lidar_rot_to_IMU*p_body + Lidar_offset_to_IMU) + state.pos_end);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity*10000;
}

void LaserMapping::pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state.rot_end * (Lidar_rot_to_IMU*p_body + Lidar_offset_to_IMU) + state.pos_end);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

bool LaserMapping::sync_packages(LidarMeasureGroup &meas){
    if ((lidar_buffer.empty() && img_buffer.empty())) {
        return false;
    }

    // 刚处理完一帧lidar数据，删除缓存区所有观测
    if (meas.is_lidar_end)
    {
        meas.measures.clear();
        meas.is_lidar_end = false;
    }

    // step1 存入lidar数据
    if (!lidar_pushed) { // 标记一帧雷达是否处理完成 If not in lidar scan, need to generate new meas
        if (lidar_buffer.empty()) {
            // ROS_ERROR("out sync");
            return false;
        }
        meas.lidar = lidar_buffer.front(); // push the firsrt lidar topic
        if(meas.lidar->points.size() <= 1)
        {
            mtx_buffer.lock();
            if (img_buffer.size()>0) // temp method, ignore img topic when no lidar points, keep sync
            {
                lidar_buffer.pop_front();
                img_buffer.pop_front();
            }
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            // ROS_ERROR("out sync");
            return false;
        }
        sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list); // sort by sample timestamp
        meas.lidar_beg_time = time_buffer.front(); // generate lidar_beg_time
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time and trans to sec
        lidar_pushed = true; // flag
    }

    // step2 存入imu数据
    struct MeasureGroup m;

    // step2.1 如果当前不存在image数据，或图像帧在当前lidar帧后
    if (img_buffer.empty() || (img_time_buffer.front()>lidar_end_time) )
    { // has img topic, but img topic timestamp larger than lidar end time, process lidar topic.
        if (last_timestamp_imu < lidar_end_time+0.02) 
        {
            // ROS_ERROR("out sync");
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time<lidar_end_time))) 
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if(imu_time > lidar_end_time) break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        lidar_pushed = false; // sync one whole lidar scan.
        meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
        meas.measures.push_back(m);
    }
    else 
    {
        double img_start_time = img_time_buffer.front(); // process img topic, record timestamp
        if (last_timestamp_imu < img_start_time) 
        {
            // ROS_ERROR("out sync");
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        m.img_offset_time = img_start_time - meas.lidar_beg_time; // record img offset time, it shoule be the Kalman update timestamp.
        m.img = img_buffer.front();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time<img_start_time))) 
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if(imu_time > img_start_time) break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        meas.is_lidar_end = false; // has img topic in lidar scan, so flag "is_lidar_end=false" 
        meas.measures.push_back(m);
    }
    // ROS_ERROR("out sync");
    return true;
}

bool LaserMapping::InitROS(ros::NodeHandle &nh){
    image_transport::ImageTransport it(nh);
    readParameters(nh);
    pcl_wait_pub->clear();

    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, &LaserMapping::livox_pcl_cbk, this) : \
        nh.subscribe(lid_topic, 200000, &LaserMapping::standard_pcl_cbk, this);
    sub_imu = nh.subscribe(imu_topic, 200000, &LaserMapping::imu_cbk, this);
    string::size_type idx;
    idx = img_topic.find("compressed");     // 判断image是否为压缩过的话题
    sub_img = idx == string::npos ? \
                nh.subscribe(img_topic, 200000, &LaserMapping::img_cbk, this) : \
                nh.subscribe(img_topic, 200000, &LaserMapping::comimg_cbk, this);

    // ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
    img_pub = it.advertise("/rgb_img", 1);
    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100);
    pubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_visual_map", 100);
    pubSubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_visual_sub_map", 100);
    pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/aft_mapped_to_init", 10);
    pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 10);
    if(loop_en) // 回环检测结果可视化
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 10);
    return true;
}

void LaserMapping::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    // cout<<"got feature"<<endl;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    // ROS_INFO("get point cloud at time: %.6f and size: %d", msg->header.stamp.toSec() - 0.1, ptr->points.size());
    printf("[ INFO ]: get point cloud at time: %.6f and size: %d.\n", msg->header.stamp.toSec(), int(ptr->points.size()));
    lidar_buffer.push_back(ptr);
    // time_buffer.push_back(msg->header.stamp.toSec() - 0.1);
    // last_timestamp_lidar = msg->header.stamp.toSec() - 0.1;
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void LaserMapping::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    printf("[ INFO ]: get point cloud at time: %.6f.\n", msg->header.stamp.toSec());
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void LaserMapping::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    //cout<<"msg_in:"<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    
    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp;

    #ifdef MINI
    // adjust for mini
    msg->linear_acceleration.x *= 200;
    msg->linear_acceleration.y *= 200;
    msg->linear_acceleration.z *= 200;
    msg->angular_velocity.x *= 200;
    msg->angular_velocity.y *= 200;
    msg->angular_velocity.z *= 200;
    #endif

    imu_buffer.push_back(msg);
    // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void LaserMapping::img_cbk(const sensor_msgs::ImageConstPtr& msg)
{
    if (!img_en) 
    {
        return;
    }
    double msg_header_time = msg->header.stamp.toSec() + delta_time;
    printf("[ INFO ]: get img at time: %.6f.\n", msg_header_time);
    if (msg_header_time < last_timestamp_img)
    {
        ROS_ERROR("img loop back, clear buffer");
        img_buffer.clear();
        img_time_buffer.clear();
    }
    mtx_buffer.lock();
    cv::Mat img;
    img = cv_bridge::toCvCopy(msg, "bgr8")->image;
    img_buffer.push_back(img);
    img_time_buffer.push_back(msg_header_time);
    last_timestamp_img = msg_header_time;

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 处理压缩后的图像信息
void LaserMapping::comimg_cbk(const sensor_msgs::CompressedImageConstPtr& msg){
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    sensor_msgs::ImagePtr image_msg = cv_ptr->toImageMsg();
    image_msg->header = msg->header;

    img_cbk(image_msg);
}

void LaserMapping::readParameters(ros::NodeHandle &nh)
{
    vector<double> extrinT(3, 0.0);
    vector<double> extrinR(9, 0.0);
    vector<double> cameraextrinT(3, 0.0);
    vector<double> cameraextrinR(9, 0.0);
    int grid_size, MIN_IMG_COUNT;
    int patch_size;
    double outlier_threshold, ncc_thre;
    double IMG_POINT_COV;
    double cam_fx, cam_fy, cam_cx, cam_cy;
    bool ncc_en;
    double gyr_cov_scale, acc_cov_scale;
    double filter_size_surf_min, filter_size_map_min;
    int ivox_nearby_type;
    
    // step1 参数读取
    nh.param<int>("dense_map_enable",dense_map_en,1);
    nh.param<int>("img_enable",img_en,1);
    nh.param<int>("lidar_enable",lidar_en,1);
    nh.param<int>("gnss/gnss_en", bgnss_en, 0);
    nh.param<int>("loop_enable", loop_en, 0);
    nh.param<int>("debug", debug, 0);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<bool>("ncc_en",ncc_en,false);
    nh.param<int>("min_img_count",MIN_IMG_COUNT,1000);    
    nh.param<double>("laserMapping/cam_fx",cam_fx, 400);
    nh.param<double>("laserMapping/cam_fy",cam_fy, 400);
    nh.param<double>("laserMapping/cam_cx",cam_cx, 300);
    nh.param<double>("laserMapping/cam_cy",cam_cy, 300);
    nh.param<double>("laser_point_cov",LASER_POINT_COV,0.001);
    nh.param<double>("img_point_cov",IMG_POINT_COV,10);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("camera/img_topic", img_topic,"/left_camera/image");
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<double>("mapping/gyr_cov_scale",gyr_cov_scale,1.0);
    nh.param<double>("mapping/acc_cov_scale",acc_cov_scale,1.0);

    // 预处理相关参数设置
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, 0);
    nh.param<double>("preprocess/fov", p_pre->hor_fov, 360);
    nh.param<int>("preprocess/scan_rang", p_pre->hor_pixel_num, 3600);
    nh.param<double>("preprocess/fov_min", p_pre->ver_min, -15);
    nh.param<double>("preprocess/fov_max", p_pre->ver_max, 15);
    nh.param<int>("preprocess/normal_neighbor", p_pre->normal_neighbor, 2);
    p_pre->init();

    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    Lidar_offset_to_IMU << VEC_FROM_ARRAY(extrinT);     // Til
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    Lidar_rot_to_IMU << MAT_FROM_ARRAY(extrinR);
    
    M3D Rcl; V3D Tcl;
    // 如果运行了mini.launch，则其对应的yaml参数会被存到 ROS 参数服务器中，则该值始终为1，后面再运行urbannav.launch也会用mini的外参
    // 所以只能在不同终端运行不同的程序
    nh.param<vector<double>>("camera/Pcl", cameraextrinT, vector<double>());
    nh.param<vector<double>>("camera/Rcl", cameraextrinR, vector<double>());
    Rcl << MAT_FROM_ARRAY(cameraextrinR);
    Tcl << VEC_FROM_ARRAY(cameraextrinT);

    nh.param<int>("grid_size", grid_size, 40);
    nh.param<int>("patch_size", patch_size, 4);
    nh.param<double>("outlier_threshold",outlier_threshold,100);
    nh.param<double>("ncc_thre", ncc_thre, 100);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<double>("delta_time", delta_time, 0.0);

    nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
    nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    // step2 初始化视觉相关参数
    lidar_selector = boost::make_shared<lidar_selection::LidarSelector>(grid_size, new SparseMap);
    // lidar_selector = new lidar_selection::LidarSelector(grid_size, new SparseMap);
    if(!vk::camera_loader::loadFromRosNs("laserMapping", lidar_selector->cam))
        throw std::runtime_error("Camera model not correctly specified.");
    lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
    lidar_selector->debug = debug;
    lidar_selector->patch_size = patch_size;
    lidar_selector->outlier_threshold = outlier_threshold;
    lidar_selector->ncc_thre = ncc_thre;
    lidar_selector->sparse_map->set_camera2lidar(Rcl, Tcl);
    lidar_selector->set_extrinsic(Lidar_offset_to_IMU, Lidar_rot_to_IMU);
    lidar_selector->state = &state;
    lidar_selector->state_propagat = &state_propagat;
    lidar_selector->NUM_MAX_ITERATIONS = NUM_MAX_ITERATIONS;
    lidar_selector->img_point_cov = IMG_POINT_COV;
    lidar_selector->fx = cam_fx;
    lidar_selector->fy = cam_fy;
    lidar_selector->cx = cam_cx;
    lidar_selector->cy = cam_cy;
    lidar_selector->ncc_en = ncc_en;
    lidar_selector->init();

    if(bgnss_en){
        p_gnss = std::make_shared<GNSSProcessing>(nh);
        string gnss_path; int use_ros;
        nh.param<int>("gnss/use_ros", use_ros, 0);
        if(!use_ros){
            nh.param<string>("gnss/path", gnss_path, "");
            p_gnss->readrtkresult(gnss_path);
        }
    }

    if(loop_en){
        ConfigSetting config_setting;
        read_parameters(nh, config_setting);

        // superpoint+lightglue组合参数
        Lightglue::Configuation cfg;
        nh.param<string>("lightglue/lightglue_path", cfg.lightglue_path, "");
        nh.param<string>("lightglue/extractor_path", cfg.extractor_path, "");
        nh.param<string>("lightglue/extractor_type", cfg.extractor_type, "superpoint");
        std::transform(cfg.extractor_type.begin(), cfg.extractor_type.end(), cfg.extractor_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });        // 转小写
        nh.param<float>("lightglue/threshold", cfg.threshold, 0.5);
        nh.param<int>("lightglue/image_size", cfg.image_size, 512);
        nh.param<string>("lightglue/device", cfg.device, "gpu");
        string mask_path;
        nh.param<string>("lightglue/mask_path", mask_path, "");
        cfg.mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        cv::threshold(cfg.mask, cfg.mask, 127, 1, cv::THRESH_BINARY);       // 二值化
        // cv::imshow("mask", cfg.mask);
        // cv::waitKey(0);
        p_lightglue = std::make_shared<Lightglue::LightGlueDecoupleOnnxRunner>(cfg);

        p_stdloop = std::make_shared<STDescManager>(config_setting);
        thread_loop_ = std::thread(&LaserMapping::loop_detect, this);
    }

    // step3 imu相关参数设定
    p_imu->set_extrinsic(Lidar_offset_to_IMU, Lidar_rot_to_IMU);
    p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
    p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
    p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));

    // step4 后端优化相关参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    // 运行相关参数设置
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    // ???
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";
}

bool LaserMapping::save_keyframe(){
    if (keyframe_count_ == 0){
        return true;
    }

    V3D transBetween = state.pos_end - last_state.pos_end;
    M3D rotBetween = state.rot_end.transpose() * last_state.rot_end;
    V3D rpy = rotBetween.eulerAngles(2, 1, 0);

    if(transBetween.norm() < 1.0 && abs(rpy(0)) < 0.2 && abs(rpy(1)) < 0.2 && abs(rpy(2)) < 0.2)
        return false;

    return true;
}

void LaserMapping::add_odofactor(){
    if (keyframe_count_ == 0){
        // 添加第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, common::trans2gtsamPose3(state.rot_end, state.pos_end), priorNoise));
        initialEstimate.insert(0, common::trans2gtsamPose3(state.rot_end, state.pos_end));
    } else{
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = common::trans2gtsamPose3(last_state.rot_end, last_state.pos_end);
        gtsam::Pose3 poseTo = common::trans2gtsamPose3(state.rot_end, state.pos_end);
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_count_ - 1, keyframe_count_, poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(keyframe_count_, poseTo);
    }
}

void LaserMapping::add_loopfactor(){
    if (loopindex_buffer.empty())
        return;
    
    for(int i=0; i<loopindex_buffer.size(); i++){
        int index_from = loopindex_buffer[i].first;
        int index_to = loopindex_buffer[i].second;
        gtsam::Pose3 poseBetween = loop_pose[i];
        gtsam::noiseModel::Diagonal::shared_ptr noisebetween = loop_noise[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(index_from, index_to, poseBetween, noisebetween));
    }

    loopindex_buffer.clear();
    loop_pose.clear();
    loop_noise.clear();
    bloop_closed = true;
}

void LaserMapping::save_keyframe_factor(){
    // step1 筛选关键帧
    if (this->save_keyframe() == false)
        return;
    
    this->add_odofactor();

    key_time[keyframe_count_] = LidarMeasures.last_update_time;

    if(loop_en) this->add_loopfactor();
    
    if(keyframe_count_ != 0){
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        if (bloop_closed){
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            
            bloop_closed = false;

            // save gtsam result
            gtsam::Values optimizedValues = isam->calculateEstimate();
            std::fstream fout(DEBUG_FILE_DIR("loop_tum.txt"), std::ios::out);
            for(const auto& key_value : optimizedValues){
                if (key_value.value.dim() == 6){
                    gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
                    
                    int keyid = static_cast<int>(key_value.key);
                    double timestamp = key_time[keyid];
                    gtsam::Rot3 rotation = pose.rotation();
                    gtsam::Point3 translation = pose.translation();
                    gtsam::Quaternion q = rotation.toQuaternion();

                    fout << std::fixed << std::setprecision(6) << timestamp << " " << std::setprecision(9) 
                        << translation.x() << " " << translation.y() << " " << translation.z() << " "
                        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
                }
            }
            fout.close();
        }
        gtSAMgraph.resize(0);
        initialEstimate.clear();
    }

    last_state = state;
    int frame_id = lidar_selector->new_frame_->id_;
    keyframe_id[frame_id] = keyframe_count_;
    keyframe_count_++;
}

void LaserMapping::loop_detect(){
    size_t cloudInd = 0;        // 彩色点云帧数目，与frame id一致
    size_t keyCloudInd = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr key_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    cv::Mat K = (cv::Mat_<float>(3, 3) << lidar_selector->fx, 0, lidar_selector->cx, 0, lidar_selector->fy, lidar_selector->cy, 0, 0, 1);

    while(1){
        std::unique_lock<std::mutex> lock(m_loop_rady);

        loop_cv.wait(lock, [this] {return !key_cloud_queue.empty();});

        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = key_cloud_queue.front();
        key_cloud_queue.pop();
        lock.unlock();

        *key_cloud += *current_cloud;

        // 累积目标数目帧为一个关键帧
        if (cloudInd % p_stdloop->config_setting_.sub_frame_num_ == 0 && cloudInd != 0){

            // step1 描述子提取
            double t0,t1,t2,t3,t4,t5; //,match_start, solve_start, svd_time;
            t0 = omp_get_wtime();
            std::vector<STDesc> stds_vec;
            p_stdloop->GenerateSTDescs(key_cloud, stds_vec);
            key_cloud->clear();
            t1 = omp_get_wtime();

            // step2 回环检测
            std::pair<int, double> search_result(-1, 0);                    // 搜索到的帧id，及回环分数
            std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;     // 回环的两帧之间的位姿变换
            loop_transform.first << 0, 0, 0;
            loop_transform.second = Eigen::Matrix3d::Identity();
            std::vector<std::pair<STDesc, STDesc>> loop_std_pair;   // 匹配上的三角描述子
            if (keyCloudInd > p_stdloop->config_setting_.skip_near_num_) {
                t0 = omp_get_wtime();
                p_stdloop->SearchLoop(stds_vec, search_result, loop_transform,
                                        loop_std_pair);
                t1 = omp_get_wtime();
            }

            // step3 将三角特征添加到数据库中
            p_stdloop->AddSTDescs(stds_vec);

            if(search_result.first > 0){
                int match_frame = search_result.first;

                std::fstream fout(DEBUG_FILE_DIR("loop_result.txt"), std::ios::app);
                // fout << "[Loop Detection] triggle loop: " << keyCloudInd << "--"
                //     << match_frame << ", score:" << search_result.second
                //     << std::endl;
                // fout.close();
                
                // step4.1 通过关键帧计算匹配帧
                int sub_frame_num = p_stdloop->config_setting_.sub_frame_num_;
                int src_id, match_id;
                for(size_t j = 1; j <= sub_frame_num; ++j){
                    src_id = cloudInd + j - sub_frame_num;
                    if(keyframe_id.count(src_id)) break;
                }

                for(size_t j = 1; j <= sub_frame_num; ++j){
                    match_id = match_frame * sub_frame_num + j;
                    if(keyframe_id.count(match_id)) break;
                }
                if(!keyframe_id.count(src_id) || !keyframe_id.count(match_id)) continue;

                // todo: 使用DBOW/视觉帧间配准？查看两帧之间匹配是否准确及帧间变化量
                cv::Mat img_cur = lidar_selector->imgs_[src_id];
                cv::Mat img_match = lidar_selector->imgs_[match_id];

                std::vector<cv::Point2f> points1, points2;
                int points_num = 0;
                auto match_points = p_lightglue->InferenceImage(img_cur, img_match, points_num);
                points1 = match_points.first;
                points2 = match_points.second;

                #ifdef SAVE_PLY
                cv::Mat img_matches;
                cv::Mat color_cur, color_match;
                cv::cvtColor(img_cur, color_cur, cv::COLOR_GRAY2BGR);
                cv::cvtColor(img_match, color_match, cv::COLOR_GRAY2BGR);
                std::vector<cv::Mat> imagesPair = {color_cur, color_match};
                std::vector<std::string> titlePair = {"srcImage", "destImage"};
                img_matches = common::plotImages(imagesPair, match_points, titlePair);
                std::stringstream filename;
                filename << "/media/zxq/T5/01graduate/03result/matches/02lightglue/" << keyCloudInd << "_" << match_frame << ".jpg";
                cv::imwrite(filename.str(), img_matches);
                #endif

                double key_rate = (double)points1.size() / (double)points_num;
                // fout << "[Loop Image] : match size: " << points1.size() << " vs " << points_num << ": " << key_rate << " " << search_result.second << std::endl;
                // fout.close();

                if(key_rate < 0.2){     // 按照匹配点及提取点的比例来判断是否为关键帧
                    fout << std::fixed << std::setprecision(6) << key_time[keyframe_id[src_id]] << " " << key_time[keyframe_id[match_id]] << " 0" << std::endl;
                    fout.close();
                    continue;
                }
                fout << std::fixed << std::setprecision(6) << key_time[keyframe_id[src_id]] << " " << key_time[keyframe_id[match_id]] << " 1" << std::endl;
                fout.close();

                cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC);
                cv::Mat R, t;
                cv::recoverPose(E, points1, points2, K, R, t);

                gtsam::Vector Vector6(6);
                Vector6 << 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1;
                // Vector6 << search_result.second, search_result.second, search_result.second,
                //         search_result.second, search_result.second, search_result.second;
                gtsam::noiseModel::Diagonal::shared_ptr constraint_noise = gtsam::noiseModel::Diagonal::Variances(Vector6);
                
                // todo: 计算回环约束后，计算相对位姿
                src_id = keyframe_id[src_id];
                match_id = keyframe_id[match_id];
                loopindex_buffer.push_back(std::make_pair(src_id, match_id));
                loop_pose.push_back(common::Mat2gtsamPose3(R, t));
                loop_noise.push_back(constraint_noise);
            }
            ++keyCloudInd;
        }
        cloudInd++;
    }
}

// void LaserMapping::publish_loop_marker(std::vector<std::pair<int, int>> &loop_container)
// {
//     if(loop_container.empty())
//         return;
    
//     visualzation_msgs::MarkerArray marker_array;
//     visualzation_msgs::Marker markerNode;
//     markerNode.header.frame_id = "camera_init";
//     markerNode.action = visualzation_msgs::Marker::ADD;
//     markerNode.type = visualzation_msgs::Marker::SPHERE_LIST;
//     markerNode.ns = "loop_nodes";
//     markerNode.id = 0;
//     markerNode.pose.orientation.w = 1;
//     markerNode.scale.x = 0.3;
//     markerNode.scale.y = 0.3;
//     markerNode.scale.z = 0.3;
//     markerNode.color.r = 0;
//     markerNode.color.g = 0.8;
//     markerNode.color.b = 1;
//     markerNode.color.a = 1;

//     // 闭环边
//     visualization_msgs::Marker markerEdge;
//     markerEdge.header.frame_id = "camera_init";
//     markerEdge.action = visualization_msgs::Marker::ADD;
//     markerEdge.type = visualization_msgs::Marker::LINE_LIST;
//     markerEdge.ns = "loop_edges";
//     markerEdge.id = 1;
//     markerEdge.pose.orientation.w = 1;
//     markerEdge.scale.x = 0.1;
//     markerEdge.color.r = 0.9;
//     markerEdge.color.g = 0.9;
//     markerEdge.color.b = 0;
//     markerEdge.color.a = 1;
    
//     // 没存pose 还不好可视化！
//     // for(auto &loop_pair : loop_container)
    
//     pubLoopConstraintEdge.publish(marker_array);
// }

void LaserMapping::publish_frame_world_rgb()
{
    uint size = pcl_wait_pub->points.size();
    PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
    if(img_en)
    {
        laserCloudWorldRGB->clear();
        cv::Mat img_rgb = lidar_selector->img_rgb;
        // cv::imwrite("/home/zxq/Documents/02base/fastlivo/"+std::to_string(frame_count_)+".jpg", img_rgb);

        for_each(pcl_wait_pub->points.begin(), pcl_wait_pub->points.end(), [&](const PointType& p){
            PointTypeRGB pointRGB;
            pointRGB.x =  p.x;
            pointRGB.y =  p.y;
            pointRGB.z =  p.z;

            V3D p_w(p.x, p.y, p.z);   // 世界坐标系下坐标
            V3D p_cam(lidar_selector->new_frame_->w2f(p_w));

            if(p_cam(2) > 0){
                V2D pc(lidar_selector->new_frame_->w2c(p_w));
                if (lidar_selector->new_frame_->cam_->isInFrame(pc.cast<int>(),0))
                {
                    V3F pixel = lidar_selector->getpixel(img_rgb, pc);
                    pointRGB.r = pixel[2];
                    pointRGB.g = pixel[1];
                    pointRGB.b = pixel[0];
                    laserCloudWorldRGB->push_back(pointRGB);
                }
            }
        });
    }

    if (1)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        if (img_en)
        {
            pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);

            if(!laserCloudWorldRGB->points.empty()){
                pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::copyPointCloud(*laserCloudWorldRGB, *current_cloud);
                
                #ifdef SAVE_PLY
                std::stringstream filename;
                filename << "/home/zxq/Documents/03data/plyfile/" << std::fixed << std::setprecision(0) << LidarMeasures.last_update_time * 1e6 << ".ply";
                pcl::io::savePLYFileBinary(filename.str(), *laserCloudWorldRGB);
                #endif

                std::unique_lock<std::mutex> lock(m_loop_rady);
                key_cloud_queue.push(current_cloud);
                lock.unlock();
                loop_cv.notify_all();   // 唤醒正在等待的线程
            }
        }
        else
        {
            pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        }
        laserCloudmsg.header.stamp = ros::Time().fromSec(LidarMeasures.last_update_time);//ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mtx_buffer_pointcloud.unlock();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en) *pcl_wait_save += *laserCloudWorldRGB;          
}

void LaserMapping::publish_odometry()
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = ros::Time().fromSec(LidarMeasures.last_update_time);
    odomAftMapped.pose.pose.position.x = state.pos_end(0);
    odomAftMapped.pose.pose.position.y = state.pos_end(1);
    odomAftMapped.pose.pose.position.z = state.pos_end(2);
    odomAftMapped.pose.pose.orientation.x = geoQuat.x;
    odomAftMapped.pose.pose.orientation.y = geoQuat.y;
    odomAftMapped.pose.pose.orientation.z = geoQuat.z;
    odomAftMapped.pose.pose.orientation.w = geoQuat.w;
    pubOdomAftMapped.publish(odomAftMapped);

    // 输出里程计位姿结果 TUM格式
    pathout << std::fixed << std::setprecision(6) << LidarMeasures.last_update_time << " " << std::setprecision(9) //
            << state.pos_end(0) << " " << state.pos_end(1) << " " << state.pos_end(2) << " " //
            << geoQuat.x << " " << geoQuat.y << " " << geoQuat.z << " " << geoQuat.w << std::endl;
}

// 可视化图像上的点
void LaserMapping::publish_visual_world_sub_map()
{
    PointCloudXYZI::Ptr laserCloudFullRes(sub_map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size==0) return;

    PointCloudXYZI::Ptr sub_pcl_visual_wait_pub(new PointCloudXYZI());
    *sub_pcl_visual_wait_pub = *laserCloudFullRes;
    if (1)//if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*sub_pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubSubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mtx_buffer_pointcloud.unlock();
}

void LaserMapping::publish_frame_world()
{
    uint size = pcl_wait_pub->points.size();
    if (1)//if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;

        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        
        laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mtx_buffer_pointcloud.unlock();
    if (pcd_save_en) *pcl_wait_save_lidar += *pcl_wait_pub;
}

// 显示雷达中的有效约束点
void LaserMapping::publish_effect_world()
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

// 输出定位轨迹
void LaserMapping::publish_path()
{
    msg_body_pose.pose.position.x = state.pos_end(0);
    msg_body_pose.pose.position.y = state.pos_end(1);
    msg_body_pose.pose.position.z = state.pos_end(2);
    msg_body_pose.pose.orientation.x = geoQuat.x;
    msg_body_pose.pose.orientation.y = geoQuat.y;
    msg_body_pose.pose.orientation.z = geoQuat.z;
    msg_body_pose.pose.orientation.w = geoQuat.w;

    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
}

