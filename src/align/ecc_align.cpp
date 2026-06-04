#include "align/ecc_align.hpp"
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace fs_core
{

    ECCAlign::ECCAlign(int iterations, double eps)
        : number_of_iterations_(iterations), termination_eps_(eps) {}

    ImageType ECCAlign::align(const ImageType &target, const ImageType &reference, cv::Rect &global_roi)
    {
        cv::UMat gray_target, gray_ref;

        if (target.channels() == 3)
        {
            cv::cvtColor(target, gray_target, cv::COLOR_BGR2GRAY);
            cv::cvtColor(reference, gray_ref, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray_target = target;
            gray_ref = reference;
        }

        // 1. Definição do Fator de Escala (0.25 reduz a área e memória em 16x)
        const double scale = 0.25;
        cv::UMat small_target, small_ref;

        // 2. Downscale rápido priorizando preservação de área
        cv::resize(gray_target, small_target, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(gray_ref, small_ref, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                  number_of_iterations_, termination_eps_);

        cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);

        try
        {
            cv::findTransformECC(small_target, small_ref, warp_matrix,
                                 cv::MOTION_AFFINE, criteria);

            warp_matrix.at<float>(0, 2) /= scale;
            warp_matrix.at<float>(1, 2) /= scale;
        }
        catch (const cv::Exception &e)
        {
            return target; // Se falhar, mantém o ROI global intacto
        }

        // Usamos a matriz M para transformar as coordenadas dos limites da imagem atual.
        float m00 = warp_matrix.at<float>(0, 0), m01 = warp_matrix.at<float>(0, 1), m02 = warp_matrix.at<float>(0, 2);
        float m10 = warp_matrix.at<float>(1, 0), m11 = warp_matrix.at<float>(1, 1), m12 = warp_matrix.at<float>(1, 2);
        int w = target.cols, h = target.rows;

        // Função lambda para projetar os cantos no referencial da imagem base
        auto transformPoint = [&](float x, float y)
        {
            return cv::Point2f(m00 * x + m01 * y + m02, m10 * x + m11 * y + m12);
        };

        cv::Point2f pt0 = transformPoint(0, 0); // Superior esquerdo
        cv::Point2f pt1 = transformPoint(w, 0); // Superior direito
        cv::Point2f pt2 = transformPoint(w, h); // Inferior direito
        cv::Point2f pt3 = transformPoint(0, h); // Inferior esquerdo

        // Calculamos o "Inner Bounding Box" (os limites mais seguros)
        int start_x = std::max({0, (int)pt0.x, (int)pt3.x});
        int start_y = std::max({0, (int)pt0.y, (int)pt1.y});
        int end_x = std::min({w, (int)pt1.x, (int)pt2.x});
        int end_y = std::min({h, (int)pt2.y, (int)pt3.y});

        cv::Rect safe_rect(start_x, start_y, std::max(0, end_x - start_x), std::max(0, end_y - start_y));

        // Acumulamos o menor retângulo comum a todas as imagens processadas até agora
        global_roi &= safe_rect;

        ImageType aligned_image;

        cv::warpAffine(target, aligned_image, warp_matrix, reference.size(),
                       cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);

        return aligned_image;
    }

}