// An example showing TEASER++ registration with FPFH features with the Stanford bunny model
#include <chrono>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/matcher.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

// Macro constants for generating noise and outliers
#define NOISE_BOUND 0.2

inline double getAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
    return std::abs(std::acos(fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
}

template<typename PointT>
void visualizeCloud(typename pcl::PointCloud<PointT>::Ptr source, typename pcl::PointCloud<PointT>::Ptr target) {
    pcl::visualization::PCLVisualizer viewer("");
    viewer.setBackgroundColor(255, 255, 255);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> sourceColor(source, 255, 180, 0);
    viewer.addPointCloud<PointT>(source, sourceColor, "source");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> targetColor(target, 0, 166, 237);
    viewer.addPointCloud<PointT>(target, targetColor, "target");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
    // set camera position at the center of the point source
    viewer.setCameraPosition(source->points[source->size() / 2].x,
                             source->points[source->size() / 2].y,
                             source->points[source->size() / 2].z,
                             0, 0, 1);
    viewer.spin();
    viewer.close();
}

template<typename T>
void voxelize(
        const boost::shared_ptr<pcl::PointCloud<T>> srcPtr, boost::shared_ptr<pcl::PointCloud<T>> dstPtr,
        double voxelSize) {
    static pcl::VoxelGrid<T> voxel_filter;
    voxel_filter.setInputCloud(srcPtr);
    voxel_filter.setLeafSize(voxelSize, voxelSize, voxelSize);
    voxel_filter.filter(*dstPtr);
}

std::vector<std::pair<int, int>> generate_correspondences(
        teaser::PointCloud &src_cloud_teaser,
        teaser::PointCloud &tgt_cloud_teaser,
        double normal_search_radius = 1.0,
        double fpfh_search_radius = 2.5) {

    // Compute FPFH
    teaser::FPFHEstimation fpfh;
    auto obj_descriptors = fpfh.computeFPFHFeatures(src_cloud_teaser, normal_search_radius, fpfh_search_radius);
    auto scene_descriptors = fpfh.computeFPFHFeatures(tgt_cloud_teaser, normal_search_radius, fpfh_search_radius);

    teaser::Matcher matcher;
    auto correspondences = matcher.calculateCorrespondences(
            src_cloud_teaser, tgt_cloud_teaser, *obj_descriptors, *scene_descriptors, true, true, false, 0.95);
    return correspondences;
}

Eigen::Matrix4d solve_correspondences(
        teaser::PointCloud &src_cloud_teaser,
        teaser::PointCloud &tgt_cloud_teaser,
        const std::vector<std::pair<int, int>> &correspondences,
        double noise_bound = 0.4) {
    // Run TEASER++ registration
    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = noise_bound;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src_cloud_teaser, tgt_cloud_teaser, correspondences);

    auto solution = solver.getSolution();
    Eigen::Matrix4d estimated_transform = Eigen::Matrix4d::Identity();
    estimated_transform.topLeftCorner(3, 3) = solution.rotation;
    estimated_transform.block<3, 1>(0, 3) = solution.translation;
    return estimated_transform;
}

Eigen::Matrix4d fpfh_teaser(const pcl::PointCloud<pcl::PointXYZ>::Ptr &src_cloud,
                            const pcl::PointCloud<pcl::PointXYZ>::Ptr &tgt_cloud) {

    // 降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_ds(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_ds(new pcl::PointCloud<pcl::PointXYZ>);
    double voxel_size = 0.5;
    voxelize(src_cloud, src_ds, voxel_size);
    voxelize(tgt_cloud, tgt_ds, voxel_size);

    // 转换为 TEASER++ 格式
    teaser::PointCloud src_cloud_teaser, tgt_cloud_teaser;
    for (size_t i = 0; i < src_ds->size(); ++i) {
        const pcl::PointXYZ &p = src_ds->points[i];
        src_cloud_teaser.push_back({p.x, p.y, p.z});
    }
    for (size_t i = 0; i < tgt_ds->size(); ++i) {
        const pcl::PointXYZ &p = tgt_ds->points[i];
        tgt_cloud_teaser.push_back({p.x, p.y, p.z});
    }

    // 生成 correspondences
    auto correspondences = generate_correspondences(src_cloud_teaser, tgt_cloud_teaser, 1.0, 2.5);

    // TEASER++
    return solve_correspondences(src_cloud_teaser, tgt_cloud_teaser, correspondences, 0.4);
}

int main() {
    // Load the .ply file
    teaser::PLYReader reader;

    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile("./example_data/outdoor/source.pcd", *src_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile("./example_data/outdoor/target.pcd", *tgt_cloud);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Eigen::Matrix4d estimated_transform = fpfh_teaser(src_cloud, tgt_cloud);
    Eigen::Matrix3d rotation = estimated_transform.block<3, 3>(0, 0);
    Eigen::Vector3d translation = estimated_transform.block<3, 1>(0, 3);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Compare results
    Eigen::Matrix4d T;
    T << 0.999998, -0.00206442, 0.000848199, 28.5538,
            0.00206406, 0.999998, 0.000423864, 0.20279,
            -0.000849072, -0.000422113, 1, 0.130061,
            0, 0, 0, 1;

    std::cout << "=====================================" << std::endl;
    std::cout << "          TEASER++ Results           " << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Expected rotation: " << std::endl;
    std::cout << T.topLeftCorner(3, 3) << std::endl;
    std::cout << "Estimated rotation: " << std::endl;
    std::cout << rotation << std::endl;
    std::cout << "Error (rad): " << getAngularError(T.topLeftCorner(3, 3), rotation)
              << std::endl;
    std::cout << std::endl;
    std::cout << "Expected translation: " << std::endl;
    std::cout << T.topRightCorner(3, 1) << std::endl;
    std::cout << "Estimated translation: " << std::endl;
    std::cout << translation << std::endl;
    std::cout << "Error (m): " << (T.topRightCorner(3, 1) - translation).norm() << std::endl;
    std::cout << std::endl;
    std::cout << "Time taken (s): "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                 1000000.0
              << std::endl;

    // Visualize results
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src_cloud, *src_transformed, estimated_transform);
    visualizeCloud<pcl::PointXYZ>(src_transformed, tgt_cloud);
}
