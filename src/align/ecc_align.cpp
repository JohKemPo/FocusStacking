#include "align/ecc_align.hpp"
#include <stdexcept>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <spdlog/spdlog.h>

namespace fs_core
{

    ECCAlign::ECCAlign(int iterations, double eps)
        : number_of_iterations_(iterations), termination_eps_(eps) {}

    cv::Mat ECCAlign::estimateTransform(const ImageType &from, const ImageType &to)
    {
        cv::UMat gray_from, gray_to;

        if (from.channels() == 3)
        {
            cv::cvtColor(from, gray_from, cv::COLOR_BGR2GRAY);
            cv::cvtColor(to, gray_to, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray_from = from;
            gray_to = to;
        }

        // 1. Downscale para reduzir custo/memoria (0.25 => area/memoria 16x menor).
        const double scale = 0.25;
        cv::UMat small_from, small_to;
        cv::resize(gray_from, small_from, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(gray_to, small_to, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                  number_of_iterations_, termination_eps_);

        // findTransformECC(template, input, W) resolve  template(x) ~= input(W*x),
        // ou seja W mapeia coordenadas de `template` -> `input`. Com template=from e
        // input=to obtemos diretamente W: from -> to (direcao usada no warp do pipeline).
        cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);

        double cc = 0.0;
        try
        {
            cc = cv::findTransformECC(small_from, small_to, warp_matrix,
                                      cv::MOTION_AFFINE, criteria);
        }
        catch (const cv::Exception &e)
        {
            return cv::Mat();
        }

        // Convergencia de baixa correlacao => alinhamento nao confiavel.
        const double min_correlation = 0.6;
        if (!std::isfinite(cc) || cc < min_correlation)
        {
            spdlog::warn("    [Align] Par rejeitado: correlacao ECC baixa ({:.3f} < {:.2f}).", cc, min_correlation);
            return cv::Mat();
        }

        // Recompoe a translacao para a resolucao original (a parte linear e invariante a escala).
        warp_matrix.at<float>(0, 2) /= scale;
        warp_matrix.at<float>(1, 2) /= scale;

        // Validacao do determinante da parte afim: rejeita inversao e escala extrema.
        const double det_affine =
            static_cast<double>(warp_matrix.at<float>(0, 0)) * warp_matrix.at<float>(1, 1) -
            static_cast<double>(warp_matrix.at<float>(0, 1)) * warp_matrix.at<float>(1, 0);
        if (!std::isfinite(det_affine) || det_affine < 0.5 || det_affine > 2.0)
        {
            spdlog::warn("    [Align] Par rejeitado: afim degenerada (det={:.3f}).", det_affine);
            return cv::Mat();
        }

        // Converte a afim 2x3 em homogenea 3x3 (from -> to).
        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat warp64;
        warp_matrix.convertTo(warp64, CV_64F);
        warp64.copyTo(H(cv::Rect(0, 0, 3, 2)));
        return H;
    }

}
