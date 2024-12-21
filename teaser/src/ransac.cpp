//
// Created by qzj on 12/21/24.
//

#include <vector>
#include <random>
#include <Eigen/Dense> // Changed from Eigen/Geometry to Eigen/Dense

#include "teaser/ransac.h"

namespace teaser {
Eigen::Matrix4d svdSE3(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt) {
  Eigen::Vector3d src_mean = src.rowwise().mean();
  Eigen::Vector3d tgt_mean = tgt.rowwise().mean();
  const Eigen::Matrix3Xd& src_centered = src - src_mean.replicate(1, src.cols());
  const Eigen::Matrix3Xd& tgt_centered = tgt - tgt_mean.replicate(1, tgt.cols());
  Eigen::MatrixXd H = src_centered * tgt_centered.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd R_ = svd.matrixV() * svd.matrixU().transpose();
  if (R_.determinant() < 0) {
    Eigen::MatrixXd V = svd.matrixV();
    V.col(2) *= -1;
    R_ = V * svd.matrixU().transpose();
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block(0, 0, 3, 3) = R_;
  T.block(0, 3, 3, 1) = tgt_mean - R_ * src_mean;
  return T;
}

Eigen::Matrix4d
solve_correspondences_by_ransac(teaser::PointCloud& src_cloud_teaser,
                                teaser::PointCloud& tgt_cloud_teaser,
                                const std::vector<std::pair<int, int>>& correspondences,
                                int max_iterations, double inlier_threshold) {
  int num_points = correspondences.size();
  int best_inliers = -1;
  Eigen::Matrix4d best_transform = Eigen::Matrix4d::Identity();
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, num_points - 1);

#pragma omp parallel for
  for (int iter = 0; iter < max_iterations; ++iter) {

    int i = dist(rng), j = dist(rng), k = dist(rng);
    int src_i_idx = correspondences[i].first, src_j_idx = correspondences[j].first,
        src_k_idx = correspondences[k].first, tgt_i_idx = correspondences[i].second,
        tgt_j_idx = correspondences[j].second, tgt_k_idx = correspondences[k].second;

    // Check if there are overlaps in the indices
    if (src_i_idx == src_j_idx || src_i_idx == src_k_idx || src_j_idx == src_k_idx ||
        tgt_i_idx == tgt_j_idx || tgt_i_idx == tgt_k_idx || tgt_j_idx == tgt_k_idx) {
      continue;
    }

    const teaser::PointXYZ& src_i = src_cloud_teaser[src_i_idx];
    const teaser::PointXYZ& tgt_i = tgt_cloud_teaser[tgt_i_idx];
    const teaser::PointXYZ& src_j = src_cloud_teaser[src_j_idx];
    const teaser::PointXYZ& tgt_j = tgt_cloud_teaser[tgt_j_idx];
    const teaser::PointXYZ& src_k = src_cloud_teaser[src_k_idx];
    const teaser::PointXYZ& tgt_k = tgt_cloud_teaser[tgt_k_idx];

    double src_ij_dist = (src_i - src_j).norm(), src_ik_dist = (src_i - src_k).norm(),
           src_jk_dist = (src_j - src_k).norm();
    double tgt_ij_dist = (tgt_i - tgt_j).norm(), tgt_ik_dist = (tgt_i - tgt_k).norm(),
           tgt_jk_dist = (tgt_j - tgt_k).norm();
    double scale = 0.95;
    // Check if the distances between the points are within translation_resolution
    if (src_ij_dist < tgt_ij_dist * scale || tgt_ij_dist < src_ij_dist * scale ||
        src_ik_dist < tgt_ik_dist * scale || tgt_ik_dist < src_ik_dist * scale ||
        src_jk_dist < tgt_jk_dist * scale || tgt_jk_dist < src_jk_dist * scale) {
      continue;
    }
    Eigen::Matrix3Xd P(3, 3), Q(3, 3);
    P.col(0) = Eigen::Vector3d(src_i.x, src_i.y, src_i.z);
    P.col(1) = Eigen::Vector3d(src_j.x, src_j.y, src_j.z);
    P.col(2) = Eigen::Vector3d(src_k.x, src_k.y, src_k.z);
    Q.col(0) = Eigen::Vector3d(tgt_i.x, tgt_i.y, tgt_i.z);
    Q.col(1) = Eigen::Vector3d(tgt_j.x, tgt_j.y, tgt_j.z);
    Q.col(2) = Eigen::Vector3d(tgt_k.x, tgt_k.y, tgt_k.z);

    Eigen::Matrix4d transform_candidate = svdSE3(P, Q);
    int inliers = 0;
    for (int n = 0; n < num_points; ++n) {
      const teaser::PointXYZ& src_point = src_cloud_teaser[correspondences[n].first];
      const teaser::PointXYZ& tgt_point = tgt_cloud_teaser[correspondences[n].second];
      const Eigen::Vector3d src(src_point.x, src_point.y, src_point.z);
      const Eigen::Vector3d tgt(tgt_point.x, tgt_point.y, tgt_point.z);
      Eigen::Vector3d transformed_src = (transform_candidate * src.homogeneous()).head<3>();
      if ((transformed_src - tgt).norm() < inlier_threshold) {
        ++inliers;
      }
    }

#pragma omp critical
    {
      if (inliers > best_inliers && inliers >= 3) {
        double inlier_ratio = static_cast<double>(inliers) / num_points;
        // update the max iterations using the inlier ratio
        max_iterations = std::min(max_iterations, static_cast<int>(log(1 - 0.999) / log(1 - pow(inlier_ratio, 3))));
        max_iterations = std::max(max_iterations, 1000);
        best_inliers = inliers;
        best_transform = transform_candidate;
      }
    }
  }
  return best_transform;
}
} // namespace teaser