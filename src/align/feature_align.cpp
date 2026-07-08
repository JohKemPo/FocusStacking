#include "align/feature_align.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace fs_core {

FeatureAlign::FeatureAlign(FeatureType type, int max_features)
    : type_(type), max_features_(max_features) {}

cv::Ptr<cv::Feature2D> FeatureAlign::createDetector() const {
    switch (type_) {
        case FeatureType::KAZE:
            return cv::KAZE::create();
        case FeatureType::SIFT:
            return cv::SIFT::create(max_features_);
        case FeatureType::ORB:
            return cv::ORB::create(max_features_);
        default:
            return cv::KAZE::create();
    }
}

ImageType FeatureAlign::align(const ImageType &target, const ImageType &reference, cv::Rect &global_roi) {
    cv::UMat gray_target, gray_ref;

    // 1. Conversao para tons de cinza preservando a pipeline transparente (UMat)
    if (target.channels() == 3) {
        cv::cvtColor(target, gray_target, cv::COLOR_BGR2GRAY);
        cv::cvtColor(reference, gray_ref, cv::COLOR_BGR2GRAY);
    } else {
        gray_target = target;
        gray_ref = reference;
    }

    // 2. Deteccao e descricao de features
    auto detector = createDetector();
    std::vector<cv::KeyPoint> kp_target, kp_ref;
    cv::UMat des_target, des_ref;

    detector->detectAndCompute(gray_target, cv::noArray(), kp_target, des_target);
    detector->detectAndCompute(gray_ref, cv::noArray(), kp_ref, des_ref);

    // Protecao contra imagens sem textura relevante
    if (des_target.empty() || des_ref.empty()) {
        return target;
    }

    // 3. Selecao do Matcher apropriado com base no espaco do descritor (Binario vs Real)
    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (type_ == FeatureType::ORB) {
        // ORB utiliza descritores binarios (BRIEF), exigindo distancia Hamming
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    } else {
        // SIFT e KAZE utilizam vetores de ponto flutuante, exigindo distancia L2 (Euclidiana)
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(des_target, des_ref, knn_matches, 2);

    // 4. Filtragem por Ratio Test de Lowe (Preservando o limiar de 0.75 do script Python)
    std::vector<cv::Point2f> pts_target, pts_ref;
    const double ratio_thresh = 0.75;

    for (const auto& match : knn_matches) {
        if (match.size() == 2 && match[0].distance < ratio_thresh * match[1].distance) {
            pts_target.push_back(kp_target[match[0].queryIdx].pt);
            pts_ref.push_back(kp_ref[match[0].trainIdx].pt);
        }
    }

    // Sao necessarios no minimo 4 pares de pontos correspondentes coplanares para calcular a Homografia
    if (pts_target.size() < 4) {
        return target; 
    }

    // 5. Calculo da Matriz de Homografia usando RANSAC para rejeicao de outliers
    cv::Mat H = cv::findHomography(pts_target, pts_ref, cv::RANSAC, 3.0);
    if (H.empty()) {
        return target;
    }

    // 6. Calculo do Inner Bounding Box seguro para o recorte (Crop)
    std::vector<cv::Point2f> corners = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(target.cols), 0),
        cv::Point2f(static_cast<float>(target.cols), static_cast<float>(target.rows)),
        cv::Point2f(0, static_cast<float>(target.rows))
    };
    
    std::vector<cv::Point2f> transformed_corners;
    cv::perspectiveTransform(corners, transformed_corners, H);

    int start_x = std::max({0, static_cast<int>(transformed_corners[0].x), static_cast<int>(transformed_corners[3].x)});
    int start_y = std::max({0, static_cast<int>(transformed_corners[0].y), static_cast<int>(transformed_corners[1].y)});
    int end_x = std::min({reference.cols, static_cast<int>(transformed_corners[1].x), static_cast<int>(transformed_corners[2].x)});
    int end_y = std::min({reference.rows, static_cast<int>(transformed_corners[2].y), static_cast<int>(transformed_corners[3].y)});

    cv::Rect safe_rect(start_x, start_y, std::max(0, end_x - start_x), std::max(0, end_y - start_y));
    
    // Atualiza a guilhotina global com a intersecao historica mais restritiva
    global_roi &= safe_rect;

    // 7. Transformacao de perspectiva (Warping) com gradiente zero nas bordas extrapoladas
    ImageType aligned_image;
    cv::warpPerspective(target, aligned_image, H, reference.size(), 
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    return aligned_image;
}

}