#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

void find_feature_matches(
        const cv::Mat &img_1, const cv::Mat &img_2,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::KeyPoint> &keypoints_2,
        std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

// BA by g2o
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose);

// global variables
std::string first_file = "../data/1.png";
std::string second_file = "../data/2.png";
std::string depth_first_file = "../data/1_depth.png";
// std::string depth_second_file = "../data/2_depth.png";

int main(int argc, char** argv) {
    // load images
    cv::Mat img_1 = cv::imread(first_file, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(second_file, cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "There are " << matches.size() << " matching points." << std::endl;

    // construct 3D points
    cv::Mat d1 = cv::imread(depth_first_file, cv::IMREAD_UNCHANGED); // 16bits, unsigned, 1 channel
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
                                        520.9, 0, 325.1,
            0, 521.0, 249.7,
            0, 0, 1);
    std::vector<cv::Point3f> pts_3d; // img_1, 3D
    std::vector<cv::Point2f> pts_2d; // img_2, 2D
    for (cv::DMatch m:matches) {
        /**
         * uchar * 	ptr (int i0=0): Returns a pointer to the specified matrix row. <- y is row
         * Point2f pt: coordinates of the keypoints
         */
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd)); // s_i * [u_i, v_i, 1]^T, real value
        pts_2d.push_back(keypoints_2[m.trainIdx].pt); // real value
        /**
         * Q: OpenCV drawMatches - queryIdx and trainIdx
         *
         * If you call the matching function in the following order:
         * match(descriptor_for_keypoints1, descriptor_for_keypoints2, matches)
         * then queryIdx refers to keypoints1 , and trainIdx refers to keypoints2 or vice versa.
         *
         * If we repeat this list of DMatch objects, each element will have the following attributes:
         * item.distance: This attribute gives us the distance between the descriptors.
         *                A lower distance indicates a better match.
         * item.trainIdx: This attribute gives us the descriptor index in the train descriptor list,
         *                in our case, its descriptor list in img2.
         * item.queryIdx: This attribute gives us the descriptor index in the list of request descriptors,
         *                in our case, this is the list of descriptors in img1.
         * item.imgIdx: This attribute gives us the image index of the train.
         */
    }
    std::cout << "3d-2d pairs: " << pts_3d.size() << std::endl;

    // solve PnP in OpenCV
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cv::Mat r, t;
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
    cv::Mat R;
    cv::Rodrigues(r, R); // formating rotation vector to rotation matrix
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve PnP in OpenCV cost time: " << time_used.count() << " seconds." << std::endl;

    std::cout << "R = " << std::endl << R << std::endl;
    std::cout << "t = " << std::endl << t << std::endl;

    // solve Pnp by Bundle Adjustment (BA)
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    std::cout << "calling bundle adjustment by gauss newton" << std::endl;
    Sophus::SE3d pose_gn;
    t1 = std::chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve PnP by gauss newton cost time: " << time_used.count() << " seconds." << std::endl;

    std::cout << "calling bundle adjustment by g2o" << std::endl;
    Sophus::SE3d pose_g2o;
    t1 = std::chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve PnP by g2o cost time: " << time_used.count() << " seconds." << std::endl;

    return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
    // initialization
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // step 1: detect Oriented FAST keypoints
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // step 2: compute descriptors
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // step 3: matching
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // step 4: select matching points
    double min_dist = 10000, max_dist = 0;

    // compute max and min distance between two keypoints
    for (int i = 0; i < descriptors_1.rows; ++i) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    for (int j = 0; j < descriptors_1.rows; ++j) {
        if (match[j].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[j]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d, // real value, camera1
        const VecVector2d &points_2d, // real value
        const cv::Mat &K,
        Sophus::SE3d &pose) {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastcost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < points_3d.size(); ++i) {
            // camera2 coordinate: P' = (T * P)_(1:3) = [X', Y', Z']
            // Sophus::SE3d &pose
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2]; // 1/Z'
            double inv_z2 = inv_z * inv_z; // 1/(Z'^2)
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy); // reprojection

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b); // dx

        if (std::isnan(dx[0])) {
            std::cout << "reault is nan!" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastcost) {
            // cost increase, update is not good
            std::cout << "cost: " << cost << ", last cost: " << lastcost << std::endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose; // Sophus::SE3d
        lastcost = cost;

        std::cout << "iteraction " << iter << " cost = " << std::setprecision(12) << cost << std::endl;
        if (dx.norm() < 1e-6)
            break; // converge
    }

    std::cout << "pose by g-n: \n" << pose.matrix() << std::endl;
}

// vertex used in g2o BA
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // reset vertex
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    /**
     * update estimate
     * left multiplication on SE3
     *  x_k+1 = x_k + dx
     */
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen <<
            update[0], update[1], update[2],
            update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    // read and write
    virtual bool read(std::istream &in) override {}
    virtual bool write(std::ostream &out) const override {}
};

// edges used in g2o BA
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * _pos3d = pos
     * _K = K
     */
    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    // compute Error
    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d); // estimated values in camera2
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    // compute Jacobian matrix
    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi <<
            -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
            0, -fy / Z, fy * Y / Z2, fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(std::istream &in) override {}
    virtual bool write(std::ostream &out) const override {}

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose) {
    // pose is 6, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    // linear solver type
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    // gradient descent methods, GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // add vertex
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // add edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "optimization costs time: " << time_used.count() << " seconds." << std::endl;
    std::cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << std::endl;
    pose = vertex_pose->estimate();
}

/*
 * -- Max dist: 94.000000
-- Min dist: 4.000000
There are 79 matching points.
3d-2d pairs: 75
solve PnP in OpenCV cost time: 0.00478711 seconds.
R =
[0.9979059095501517, -0.05091940089119591, 0.03988747043579327;
 0.04981866254262534, 0.9983623157437967, 0.02812094175427922;
 -0.04125404886006491, -0.02607491352939112, 0.9988083912027803]
t =
[-0.1267821389545255;
 -0.00843949681832986;
 0.06034935748864372]
calling bundle adjustment by gauss newton
iteraction 0 cost = 40517.7576706
iteraction 1 cost = 410.547029116
iteraction 2 cost = 299.76468142
iteraction 3 cost = 299.763574327
pose by g-n:
   0.997905909549  -0.0509194008562   0.0398874705187   -0.126782139096
   0.049818662505    0.998362315745   0.0281209417649 -0.00843949683874
 -0.0412540489424  -0.0260749135374    0.998808391199   0.0603493575229
                0                 0                 0                 1
solve PnP by gauss newton cost time: 8.9581e-05 seconds.

calling bundle adjustment by g2o
optimization costs time: 0.000309158 seconds.
pose estimated by g2o =
    0.99790590955  -0.0509194008911   0.0398874704367   -0.126782138956
  0.0498186625425    0.998362315744   0.0281209417542 -0.00843949681823
 -0.0412540488609  -0.0260749135293    0.998808391203   0.0603493574888
                0                 0                 0                 1
solve pnp by g2o cost time: 0.000403536 seconds.
iteration= 0	 chi2= 410.547029	 time= 1.8283e-05	 cumTime= 1.8283e-05	 edges= 75	 schur= 0
iteration= 1	 chi2= 299.764681	 time= 9.875e-06	 cumTime= 2.8158e-05	 edges= 75	 schur= 0
iteration= 2	 chi2= 299.763574	 time= 9.034e-06	 cumTime= 3.7192e-05	 edges= 75	 schur= 0
iteration= 3	 chi2= 299.763574	 time= 8.933e-06	 cumTime= 4.6125e-05	 edges= 75	 schur= 0
iteration= 4	 chi2= 299.763574	 time= 8.786e-06	 cumTime= 5.4911e-05	 edges= 75	 schur= 0
iteration= 5	 chi2= 299.763574	 time= 8.766e-06	 cumTime= 6.3677e-05	 edges= 75	 schur= 0
iteration= 6	 chi2= 299.763574	 time= 8.729e-06	 cumTime= 7.2406e-05	 edges= 75	 schur= 0
iteration= 7	 chi2= 299.763574	 time= 8.724e-06	 cumTime= 8.113e-05	 edges= 75	 schur= 0
iteration= 8	 chi2= 299.763574	 time= 8.715e-06	 cumTime= 8.9845e-05	 edges= 75	 schur= 0
iteration= 9	 chi2= 299.763574	 time= 8.723e-06	 cumTime= 9.8568e-05	 edges= 75	 schur= 0

Process finished with exit code 0
*/