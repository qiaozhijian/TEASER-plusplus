//
// Created by qzj on 12/21/24.
//

#ifndef TEASERPP_RANSAC_H
#define TEASERPP_RANSAC_H

#include <Eigen/Core>
#include <teaser/registration.h>
namespace teaser {
Eigen::Matrix4d
solve_correspondences_by_ransac(teaser::PointCloud& src_cloud_teaser,
                                teaser::PointCloud& tgt_cloud_teaser,
                                const std::vector<std::pair<int, int>>& correspondences,
                                int max_iterations = 1e8, double inlier_threshold = 0.5);
}
#endif // TEASERPP_RANSAC_H
