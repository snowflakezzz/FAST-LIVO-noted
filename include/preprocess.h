#include "common_lib.h"
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE{AVIA = 1, VELO16, OUST64, XT32}; //{1, 2, 3}
enum Feature{Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint};
enum Surround{Prev, Next};
enum E_jump{Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind};

struct orgtype
{
  double range;
  double dista; 
  double angle[2];
  double intersect;
  E_jump edj[2];
  Feature ftype;
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      float time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      uint8_t ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

namespace xt32_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace xt32_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(xt32_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(uint16_t, ring, ring))


// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);
  void init();

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;     // pl_surf存储所有点云（不提特征时）
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  int lidar_type, point_filter_num, N_SCANS;
  double blind;
  bool feature_enabled,given_offset_time;
  ros::Publisher pub_full, pub_surf, pub_corn;
  
  // 点云法向量提取相关参数
  double ver_min, ver_max;  // 扫描角度的左右限制 度
  double hor_fov;           // 视场角
  int hor_pixel_num;     // 每个线激光点数
  int normal_neighbor;    // 提取法向量时用多少个相邻点

  private:
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void xt32_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  void extract_normal();
  int uv2index(const int &u, const int &v);
  void index2uv(const int &index, int &u, int &v);

  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;

  double hor_resolution, ver_resolution;
  vector<double> uv_index;
  vector<double> index_uv;
  vector<Eigen::Matrix3d> sp2cart_map;  // 极坐标系转笛卡尔坐标系转换矩阵

  template<typename T>
  void get_uv(T point, int & u, int & v){
      float xy_range = std::sqrt(point.x * point.x + point.y * point.y);
      if(xy_range < 1e-3) {
          u = -1;
          v = -1;
          return;
      }

      double azimuth =  PI_M - atan2(point.y, point.x);
      double elevation = atan2(point.z, xy_range);

      int u_ = azimuth / hor_resolution;
      int v_ = ((ver_max * PI_M/180.0) - elevation) / ver_resolution;

      if(u_ < 0 || u_ >= hor_pixel_num || v_ < 0 || v_ >= N_SCANS) {
          u = -1;
          v = -1;
          return;
      }

      u = u_;
      v = v_;
      return;
  }
};