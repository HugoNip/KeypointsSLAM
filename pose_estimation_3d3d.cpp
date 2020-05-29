#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <chrono>
#include <sophus/se3.hpp>

void find_feature_matches(
        const cv::Mat &img_1, const cv::Mat &img_2,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::KeyPoint> &keypoints_2,
        std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void pose_estimation_3d3d(
        const std::vector<cv::Point3f> &pts1,
        const std::vector<cv::Point3f> &pts2,
        cv::Mat &R, cv::Mat &t);

void bundleAdjustment(
        const std::vector<cv::Point3f> &points_3d, // 3D points
        const std::vector<cv::Point3f> &points_2d,
        cv::Mat &R, cv::Mat &t);

// vertex used in g2o BA
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // sets the node to the origin (used in the multilevel stuff)
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    // left multiplication on SE3
    // update the position of the node from the parameters in v. Implement in your class!
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override {}
    virtual bool write(std::ostream &out) const override {}
};

// edge used in g2o BA
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

    virtual void computeError() override {
        const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
        _error = _measurement - pose->estimate() * _point;
    }

    virtual void linearizeOplus() override {
        VertexPose *pose = static_cast<VertexPose *>(_vertices[0]); // R^6
        Sophus::SE3d T = pose->estimate(); // R^6
        Eigen::Vector3d xyz_trans = T * _point; // R^3
        // R3x6
        // hat(.): R^6 -> R^{4x4}
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity(); // R3x3
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans); // R3x3
    }

    bool read(std::istream &in) {}
    bool write(std::ostream &out) const {}

protected:
    Eigen::Vector3d _point; // R^3
};

// global variables
std::string first_file = "../data/1.png";
std::string second_file = "../data/2.png";
std::string depth_first_file = "../data/1_depth.png";
std::string depth_second_file = "../data/2_depth.png";

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
    cv::Mat depth1 = cv::imread(depth_first_file, cv::IMREAD_UNCHANGED); // 16bits, unsigned, 1 channel
    cv::Mat depth2 = cv::imread(depth_second_file, cv::IMREAD_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts1, pts2; // img_1, 3D

    for (cv::DMatch m:matches) {
        /**
         * uchar * 	ptr (int i0=0): Returns a pointer to the specified matrix row. <- y is row
         * Point2f pt: coordinates of the keypoints
         */
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0)
            continue;
        float dd1 = d1 / 5000.0;
        float dd2 = d2 / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        pts1.push_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1)); // s_i * [u_i, v_i, 1]^T, real value
        pts2.push_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    std::cout << "3d-3d pairs: " << pts1.size() << std::endl;

    cv::Mat R, t;


    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    pose_estimation_3d3d(pts1, pts2, R, t);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "SVD costs time: " << time_used.count() << " seconds." << std::endl;


    std::cout << "ICP via SVD results: " << std::endl;
    std::cout << "R = " << R << std::endl;
    std::cout << "t = " << t << std::endl;
    std::cout << "R_inv = " << R.t() << std::endl;
    std::cout << "t_inv = " << -R.t() * t << std::endl;

    // BA
    t1 = std::chrono::steady_clock::now();
    bundleAdjustment(pts1, pts2, R, t);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "BA costs time: " << time_used.count() << " seconds." << std::endl;



    // vertify p1 = R * p2 + t
    for (int i = 0; i < 5; ++i) {
        std::cout << "p1 = " << pts1[i] << std::endl;
        std::cout << "p2 = " << pts2[i] << std::endl;
        std::cout << "(R*p2+t) = " << R * (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << std::endl;
        std::cout << std::endl;
    }

    return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
    cv::Mat descriptors_1, descriptors_2;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // step 1
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // step 2
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // step 3
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // step 4
    double min_dist = 10000, max_dist = 0;

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
    return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1,
                          const std::vector<cv::Point3f> &pts2,
                          cv::Mat &R, cv::Mat &t) {
    cv::Point3f p1, p2; // mass center

    int N = pts1.size();
    for (int i = 0; i < N; ++i) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);

    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for (int j = 0; j < N; ++j) {
        q1[j] = pts1[j] - p1;
        q2[j] = pts2[j] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int k = 0; k < N; ++k) {
        W += Eigen::Vector3d(q1[k].x, q1[k].y, q1[k].z) * Eigen::Vector3d(q2[k].x, q2[k].y, q2[k].z).transpose();
    }
    std::cout << "W = " << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    std::cout << "U = " << U << std::endl;
    std::cout << "V = " << V << std::endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) <<
            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2));

    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(
        const std::vector<cv::Point3f> &pts1,
        const std::vector<cv::Point3f> &pts2,
        cv::Mat &R, cv::Mat &t) {
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    VertexPose *pose = new VertexPose();
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    // edges
    for (size_t i = 0; i < pts1.size(); ++i) {
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(
                pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "optimization costs time: " << time_used.count() << " seconds." << std::endl;

    std::cout << std::endl << "after optimization: " << std::endl;
    std::cout << "T = \n" << pose->estimate().matrix() << std::endl;

    // convert to cv::Mat
    Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = pose->estimate().translation();
    R = (cv::Mat_<double>(3, 3) <<
                                R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2));

    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}


/*
 * -- Max dist: 94.000000
-- Min dist: 4.000000
There are 79 matching points.
3d-3d pairs: 72
W =   10.871 -1.01948  2.54771
-2.16033  3.85307 -5.77742
 3.94738 -5.79979  9.62203
U =   0.558087  -0.829399 -0.0252034
 -0.428009  -0.313755   0.847565
  0.710878   0.462228   0.530093
V =   0.617887  -0.784771 -0.0484806
 -0.399894  -0.366747   0.839989
  0.676979   0.499631   0.540434
SVD costs time: 9.808e-05 seconds.
ICP via SVD results:
R = [0.9969452351705235, 0.0598334759429696, -0.05020112774999549;
 -0.05932607556034211, 0.9981719680327525, 0.01153858709846634;
 0.05079975225724825, -0.008525103530306, 0.9986724727258676]
t = [0.1441598281917405;
 -0.06667849447794799;
 -0.03009747343724256]
R_inv = [0.9969452351705235, -0.05932607556034211, 0.05079975225724825;
 0.0598334759429696, 0.9981719680327525, -0.008525103530306;
 -0.05020112774999549, 0.01153858709846634, 0.9986724727258676]
t_inv = [-0.1461462830262246;
 0.0576744363694081;
 0.03806387978797152]
optimization costs time: 0.0003332 seconds.

after optimization:
T =
  0.996945  0.0598335 -0.0502011    0.14416
-0.0593261   0.998172  0.0115386 -0.0666785
 0.0507998 -0.0085251   0.998672 -0.0300979
         0          0          0          1
BA costs time: 0.000468802 seconds.
p1 = [-0.243698, -0.117719, 1.5848]
p2 = [-0.297211, -0.0956614, 1.6558]
(R*p2+t) = [-0.2409901495364604;
 -0.1254270500587826;
 1.609221205029395]

p1 = [0.402045, -0.341821, 2.2068]
p2 = [0.378811, -0.262859, 2.2196]
(R*p2+t) = [0.3946591022539743;
 -0.3259188829495218;
 2.20803983035825]

p1 = [-0.522843, -0.214436, 1.4956]
p2 = [-0.58581, -0.208584, 1.6052]
(R*p2+t) = [-0.532923946912698;
 -0.2216052393093164;
 1.54499035805527]

p1 = [-0.627753, 0.160186, 1.3396]
p2 = [-0.709645, 0.159033, 1.4212]
(R*p2+t) = [-0.6251478068660965;
 0.1505624195985039;
 1.351809862638435]

p1 = [0.594266, -0.0256024, 1.5332]
p2 = [0.514795, 0.0391393, 1.5332]
(R*p2+t) = [0.5827556962439571;
 -0.04046060384335358;
 1.526884519595548]

iteration= 0	 chi2= 1.816112	 time= 3.6523e-05	 cumTime= 3.6523e-05	 edges= 72	 schur= 0	 lambda= 0.000758	 levenbergIter= 1
iteration= 1	 chi2= 1.815514	 time= 1.8849e-05	 cumTime= 5.5372e-05	 edges= 72	 schur= 0	 lambda= 0.000505	 levenbergIter= 1
iteration= 2	 chi2= 1.815514	 time= 1.716e-05	 cumTime= 7.2532e-05	 edges= 72	 schur= 0	 lambda= 0.000337	 levenbergIter= 1
iteration= 3	 chi2= 1.815514	 time= 1.7532e-05	 cumTime= 9.0064e-05	 edges= 72	 schur= 0	 lambda= 0.000225	 levenbergIter= 1
iteration= 4	 chi2= 1.815514	 time= 1.7492e-05	 cumTime= 0.000107556	 edges= 72	 schur= 0	 lambda= 0.000150	 levenbergIter= 1
iteration= 5	 chi2= 1.815514	 time= 1.692e-05	 cumTime= 0.000124476	 edges= 72	 schur= 0	 lambda= 0.000299	 levenbergIter= 1
 */
