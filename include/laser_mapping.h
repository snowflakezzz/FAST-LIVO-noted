#ifndef FAST_LIVO_LASER_MAPPING_H
#define FAST_LIVO_LASER_MAPPING_H

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <image_transport/image_transport.h>
#include "IMU_Processing.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "lidar_selection.h"
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <cv_bridge/cv_bridge.h>
#include "GNSS_Processing.h"
#include "ivox3d/ivox3d.h"
#include <pcl/kdtree/kdtree_flann.h>
#include "STD/STDesc.h"
#ifdef USE_ikdtree
#include "ikd-Tree/ikd_Tree.h"
#endif

// gtsam
#include <gtsam/geometry/Rot3.h>
// #include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Factor.h>
#include "LightGlue/LightGlueDecoupleOnnxRunner.h"

#define INIT_TIME           (0.5)
#define MAXN                (360000)
#define PUBFRAME_PERIOD     (20)

class LaserMapping {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::DEFAULT, PointType>;

    LaserMapping();
    ~LaserMapping();

    bool InitROS(ros::NodeHandle &nh);

    void Run();

    void Finish();

    // ros::NodeHandler m_nh;
private:
    void readParameters(ros::NodeHandle &nh);
    void caculate_covariance(PointCloudXYZI::Ptr &cloud_in, vector<M3D> &covariances);
    
    bool sync_packages(LidarMeasureGroup &meas);

    void h_share_model(MatrixXd &HPH, VectorXd &HPL);

    void map_incremental();
    void loop_detect();

    void pointBodyToWorld(PointType const * const pi, PointType * const po);
    void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);

    // 图像回调函数，将图像存储到缓冲区img_buffer，对应时间存储到img_time_buffer
    void img_cbk(const sensor_msgs::ImageConstPtr& msg);
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in); 
    void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg); 
    void comimg_cbk(const sensor_msgs::CompressedImageConstPtr& msg);

    void publish_frame_world_rgb();
    void publish_odometry();
    void publish_visual_world_sub_map();
    void publish_frame_world();
    void publish_effect_world();
    void publish_path();

    // gtsam
    void save_keyframe_factor();
    void add_loopfactor();
    void add_odofactor();
    bool save_keyframe();

    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    shared_ptr<Preprocess> p_pre;
    shared_ptr<ImuProcess> p_imu;
    GNSSProcessing::Ptr p_gnss;
    STDescManager::Ptr p_stdloop;
    lidar_selection::LidarSelectorPtr lidar_selector;              // visual part
    Lightglue::LightGlueDecoupleOnnxRunner::Ptr p_lightglue;       // 视觉特征提取及匹配

    std::thread         thread_loop_;

    PointCloudXYZI::Ptr pcl_wait_pub;
    mutex mtx_buffer;
    condition_variable sig_buffer;
    bool flg_first_scan;

    string map_file_path, lid_topic, imu_topic, img_topic, config_file;
    Vector3d Lidar_offset_to_IMU;
    M3D Lidar_rot_to_IMU;
    int feats_down_size = 0, NUM_MAX_ITERATIONS = 0,\
        effct_feat_num = 0, time_log_counter = 0, publish_count = 0;

    //double last_timestamp_lidar,  当前接收到的最新imu数据的时间戳;
    double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;

    double filter_size_corner_min = 0, fov_deg = 0, filter_size_map_min = 0;

    double cube_len = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
    double first_img_time=-1.0;

    double kdtree_incremental_time = 0, kdtree_search_time = 0, kdtree_delete_time = 0.0;
    int kdtree_search_counter = 0, kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;;

    double copy_time = 0, readd_time = 0, fov_check_time = 0, readd_box_time = 0, delete_box_time = 0;
    double T1[MAXN], T2[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];

    double match_time = 0, solve_time = 0, solve_const_H_time = 0;

    bool lidar_pushed, flg_reset = false;
    int dense_map_en = 1;
    int img_en = 1;                 // 是否使用视觉观测
    int lidar_en = 1;               // 是否使用Lidar观测
    int bgnss_en = 1;               // 是否使用GPS
    int loop_en = 1;                // 是否进行回环检测
    int debug = 0;
    bool fast_lio_is_ready = false;
    double delta_time = 0.0;

    deque<PointCloudXYZI::Ptr>  lidar_buffer;
    deque<double>          time_buffer;
    deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
    deque<cv::Mat> img_buffer;
    deque<double>  img_time_buffer;
    vector<PointVector> Nearest_Points; 
    double LASER_POINT_COV; 
    bool flg_EKF_inited, flg_EKF_converged;
    //surf feature in map
    PointCloudXYZI::Ptr sub_map_cur_frame_point;

    PointCloudXYZI::Ptr feats_undistort;
    PointCloudXYZI::Ptr feats_down_body;      // 当前帧点云
    PointCloudXYZI::Ptr feats_down_world;     // 地图点

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    #ifdef USE_ikdtree
    KD_TREE ikdtree;        // ikdtree中存在多线程，所以必须定义为全局变量 https://github.com/hku-mars/ikd-Tree/issues/8
    #endif

    V3D euler_cur;
    V3D position_last;

    //estimator inputs and output;
    LidarMeasureGroup LidarMeasures;

    nav_msgs::Path path;
    nav_msgs::Odometry odomAftMapped;
    geometry_msgs::Quaternion geoQuat;
    geometry_msgs::PoseStamped msg_body_pose;

    PointCloudXYZRGB::Ptr pcl_wait_save;  //add save rbg map
    PointCloudXYZI::Ptr pcl_wait_save_lidar;  //add save xyzi map
    PointCloudXYZI::Ptr laserCloudOri;     // 记录用于构建与约束的点

    bool pcd_save_en = true;
    int pcd_save_interval = 20, pcd_index = 0;

    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;
    ros::Subscriber sub_img;
    ros::Publisher pubLaserCloudFullRes;
    ros::Publisher pubVisualCloud;
    ros::Publisher pubSubVisualCloud;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;
    image_transport::Publisher img_pub;
    ros::Publisher pubLoopConstraintEdge;

    // 滤波优化相关参数
    StatesGroup state;                         // 系统状态量
    StatesGroup state_propagat;                // 上一次优化后的状态量
    StatesGroup last_state;

    bool nearest_search_en;                     // 判断是否要进行最近临点搜索

    ofstream pathout;
    ofstream fout_out;

    // 多线程相关参数
    std::condition_variable loop_cv;
    std::mutex m_loop_rady;
    vector<pair<int, int>> loopindex_buffer;
    vector<gtsam::Pose3> loop_pose;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loop_noise;

    // 回环及gnss融合优化
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::ISAM2 *isam;
    gtsam::Values initialEstimate;
    int keyframe_count_ = 0;
    unordered_map<int, int> keyframe_id;         // 存储正常帧对应的关键帧id
    unordered_map<int, double> key_time;         // 存储关键帧对应的时间戳
    bool bloop_closed = false;

    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_queue;   // 用于loop detector存储关键帧点云队列
    std::vector<V3D>  weak_directions_g;
};
#endif