#ifndef LIDAR_SELECTION_H_
#define LIDAR_SELECTION_H_

#include <common_lib.h>
#include <vikit/abstract_camera.h>
#include <frame.h>
#include <map.h>
#include <feature.h>
#include <point.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <set>

namespace lidar_selection {

class LidarSelector {
  public:
    int grid_size;
    vk::AbstractCamera* cam;                      // 相机模型
    SparseMap* sparse_map;
    StatesGroup* state;
    StatesGroup* state_propagat;
    M3D Rli, Rci, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
    V3D Pli, Pci, Pcw;
    int* align_flag;
    int* grid_num;
    int* map_index;                               // 格网索引
    float* map_dist;                              // 当前帧各个格网中到光心最近的地图点
    float* map_value;                             // 存储格网shiTomasi角点评分分数
    float* patch_cache;
    float* patch_with_border_;
    int width, height, grid_n_width, grid_n_height, length;   // length格网个数
    SubSparseMap* sub_sparse_map;                 // 用于构建视觉残差的点
    double fx,fy,cx,cy;
    bool ncc_en;
    int debug, patch_size, patch_size_total, patch_size_half;
    int count_img, MIN_IMG_COUNT;
    int NUM_MAX_ITERATIONS;
    double img_point_cov, outlier_threshold, ncc_thre;
    size_t n_meas_;                //!< Number of measurements
    deque< PointPtr > map_cur_frame_;
    deque< PointPtr > sub_map_cur_frame_;         // 用于可视化
    double computeH, ekf_time;
    double ave_total = 0.0;
    int frame_count = 0;

    Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
    MatrixXd H_sub, K;

    LidarSelector(const int grid_size, SparseMap* sparse_map);

    ~LidarSelector();

    void detect(cv::Mat img, PointCloudXYZI::Ptr pg);
    float CheckGoodPoints(cv::Mat img, V2D uv);
    void addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg);
    void addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg);
    void FeatureAlignment(cv::Mat img);
    void set_extrinsic(const V3D &transl, const M3D &rot);
    void init();
    void getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level);
    void dpi(V3D p, MD(2,3)& J);
    float UpdateState(cv::Mat img, float total_residual, int level);
    double NCC(float* ref_patch, float* cur_patch, int patch_size);

    void ComputeJ(cv::Mat img);
    void reset_grid();
    void addObservation(cv::Mat img);
    void createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref);
    void getWarpMatrixAffine(
      const vk::AbstractCamera& cam,
      const Vector2d& px_ref,
      const Vector3d& f_ref,
      const double depth_ref,
      const SE3& T_cur_ref,
      const int level_ref,    // px_ref对应特征点的金字塔层级
      const int pyramid_level,
      const int halfpatch_size,
      Matrix2d& A_cur_ref);
    bool align2D(
      const cv::Mat& cur_img,
      float* ref_patch_with_border,
      float* ref_patch,
      const int n_iter,
      Vector2d& cur_px_estimate,
      int index);
    void AddPoint(PointPtr pt_new);
    int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level);
    void display_keypatch(double time);
    void updateFrameState(StatesGroup state);   // 更新当前帧状态
    V3F getpixel(cv::Mat img, V2D pc);

    void warpAffine(
      const Matrix2d& A_cur_ref,
      const cv::Mat& img_ref,
      const Vector2d& px_ref,
      const int level_ref,
      const int search_level,
      const int pyramid_level,
      const int halfpatch_size,
      float* patch);
    
    pcl::VoxelGrid<PointType> downSizeFilter;
    unordered_map<VOXEL_KEY, VOXEL_POINTS*> feat_map;     // 视觉点云地图
    unordered_map<VOXEL_KEY, float> sub_feat_map;         // 记录当前帧各个voxel块中点的数量
    unordered_map<int, Warp*> Warp_map;                   // 记录每个特征点对应的参考帧reference frame id, A_cur_ref and search_level

    vector<PointPtr> voxel_points_;
    vector<V3D> add_voxel_points_;

    cv::Mat img_cp, img_rgb;
    int img_ind = 0;      // 用于图片输出名称
    unordered_map<int, cv::Mat> imgs_;                     // 存储历史图像，用于patch块去畸变
    FramePtr new_frame_;                                   // 新图像观测帧
    Map map_;
    enum Stage {
      STAGE_FIRST_FRAME,
      STAGE_DEFAULT_FRAME
    };
    Stage stage_;
    enum CellType {
      TYPE_MAP = 1,
      TYPE_POINTCLOUD,
      TYPE_UNKNOWN
    };

  private:
    struct Candidate 
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      PointPtr pt;       
      Vector2d px;    
      Candidate(PointPtr pt, Vector2d& px) : pt(pt), px(px) {}
    };
    typedef std::list<Candidate > Cell;
    typedef std::vector<Cell*> CandidateGrid;

    /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
    struct Grid
    {
      CandidateGrid cells;
      vector<int> cell_order;
      int cell_size;
      int grid_n_cols;
      int grid_n_rows;
      int cell_length;
    };

    Grid grid_;
};
  typedef boost::shared_ptr<LidarSelector> LidarSelectorPtr;

} // namespace lidar_detection

#endif // LIDAR_SELECTION_H_