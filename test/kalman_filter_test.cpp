#include <argparse/argparse.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>

#include "kalman_filter.hpp"

static constexpr size_t N = 4096;
static constexpr float T = 4.0F;
static constexpr float dt = T / N;

template <size_t Dim>
static void assginRowVec(cv::Mat dst, cv::Matx<float, Dim, 1> src, int rowIndex) {
    for (size_t j = 0; j < Dim; j++) {
        dst.at<float>(rowIndex, j) = src(j);
    }
}

int main(int argc, char* argv[]) {
    auto t = cv::Mat({cv::Size(1, N)}, CV_32F);
    auto kf = KalmanFilter<4, 2, 1>();

    // clang-format off
    auto F = cv::Matx44f {
        1.0F, dt,   0.0F, 0.0F, 
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, dt, 
        0.0F, 0.0F, 0.0F, 1.0F
    };

    auto H = cv::Matx<float, 2, 4> {
        1.0F, 0.0F, 0.0F, 0.0F, 
        0.0F, 0.0F, 1.0F, 0.0F
    };

    auto B = cv::Vec4f {
        0.0F, 0.0F, 0.5F * dt * dt, dt
    };
    // clang-format on

    auto Q = cv::Matx44f::zeros();
    auto R = cv::Matx22f::diag({4.0F, 25.0F});
    auto P = cv::Matx44f::diag(cv::Vec4f::all(16.0F));

    // State x = [x_pos, x_velocity, y_pos, y_velocity]
    auto xInit = cv::Vec4f{0.0F, 10.0F, 0.0F, 10.0F};
    // Control u = [y_acceleration]
    auto u = cv::Matx<float, 1, 1> (-9.80665F);

    auto xInitNoised = xInit + cv::Vec4f::randn(0.0F, 4.0F);
    kf.setState(xInitNoised);
    kf.setStateCovMatrix(P);

    kf.setStateTransitionMatrix(F);
    kf.setMeasurementMatrix(H);
    kf.setControlTransitionMatrix(B);

    kf.setProcessNoiseCovMatrix(Q);
    kf.setMeasurementNoiseCovMatrix(R);

    // Groundtruth state sequence
    auto xtGt = cv::Mat(N, 4, CV_32F, 0.0F);
    // Estimated state sequence
    auto xtEst = cv::Mat(N, 4, CV_32F, 0.0F);
    // Measuremnt sequence
    auto zt = cv::Mat(N, 2, CV_32F, 0.0F);
    assginRowVec<4>(xtGt, xInit, 0);
    
    std::cout << xtGt.row(0) << std::endl;

    auto z = cv::Vec2f{xInit[0], xInit[2]};
    z[0] += cv::theRNG().gaussian(std::sqrt(R(0, 0)));
    z[1] += cv::theRNG().gaussian(std::sqrt(R(1, 1)));

    assginRowVec<2>(zt, z, 0);

    for (size_t i = 1; i < N; i++) {
        auto xPre = cv::Vec4f(xtGt.row(i - 1));
        auto x = F * xPre + B * u;
        assginRowVec<4>(xtGt, x, i);

        z[0] = x(0) + cv::theRNG().gaussian(std::sqrt(R(0, 0)));
        z[1] = x(2) + cv::theRNG().gaussian(std::sqrt(R(1, 1)));

        assginRowVec<2>(zt, z, i);
    }

    // Run kalman filter

    for (size_t i = 0; i < N; i++) {
        kf.predict(u);
        
        auto z = cv::Vec2f{zt.row(i)};
        auto xEst = kf.update(z);
        assginRowVec<4>(xtEst, xEst, i);
    }

    auto output =
        cv::FileStorage("data/kalman_test.json",
                        cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    output << "xt_gt" << xtGt;
    output << "xt_est" << xtEst;
    output << "zt" << zt;
    output.release();

    std::cout << cv::format(zt, cv::Formatter::FMT_NUMPY) << std::endl;
    return 0;
}