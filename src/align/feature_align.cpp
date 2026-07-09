#include "align/feature_align.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <vector>

namespace {

// Valida uma transformacao de similaridade (translacao, rotacao, escala uniforme)
// estimada para focus stacking. Como o modelo parcial-afim ja nao tem perspectiva
// nem cisalhamento, basta garantir que a escala seja plausivel (sem colapso nem
// explosao) e que a matriz nao esteja invertida/degenerada.
bool isSimilaritySane(const cv::Mat &M)
{
    if (M.empty() || M.rows != 2 || M.cols != 3) return false;

    const double a = M.at<double>(0, 0), b = M.at<double>(0, 1);
    const double c = M.at<double>(1, 0), d = M.at<double>(1, 1);

    // Para uma similaridade, det = escala^2. Limita a escala a [0.5x, 2x].
    const double det = a * d - b * c;
    if (!std::isfinite(det) || det < 0.25 || det > 4.0) return false;

    return true;
}

// Converte uma matriz afim 2x3 em uma homogenea 3x3 (bottom row [0 0 1]).
cv::Mat toHomogeneous(const cv::Mat &affine_2x3)
{
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat affine64;
    affine_2x3.convertTo(affine64, CV_64F);
    affine64.copyTo(H(cv::Rect(0, 0, 3, 2)));
    return H;
}

} // namespace

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

cv::Mat FeatureAlign::estimateTransform(const ImageType &from, const ImageType &to) {
    cv::UMat gray_from, gray_to;

    // 1. Conversao para tons de cinza preservando a pipeline transparente (UMat)
    if (from.channels() == 3) {
        cv::cvtColor(from, gray_from, cv::COLOR_BGR2GRAY);
        cv::cvtColor(to, gray_to, cv::COLOR_BGR2GRAY);
    } else {
        gray_from = from;
        gray_to = to;
    }

    // 2. Deteccao e descricao de features
    auto detector = createDetector();
    std::vector<cv::KeyPoint> kp_from, kp_to;
    cv::UMat des_from, des_to;

    detector->detectAndCompute(gray_from, cv::noArray(), kp_from, des_from);
    detector->detectAndCompute(gray_to, cv::noArray(), kp_to, des_to);

    // Protecao contra imagens sem textura relevante
    if (des_from.empty() || des_to.empty()) {
        return cv::Mat();
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
    matcher->knnMatch(des_from, des_to, knn_matches, 2);

    // 4. Filtragem por Ratio Test de Lowe
    std::vector<cv::Point2f> pts_from, pts_to;
    const double ratio_thresh = 0.75;

    for (const auto& match : knn_matches) {
        if (match.size() == 2 && match[0].distance < ratio_thresh * match[1].distance) {
            pts_from.push_back(kp_from[match[0].queryIdx].pt);
            pts_to.push_back(kp_to[match[0].trainIdx].pt);
        }
    }

    // Sao necessarios ao menos alguns pares para estimar a transformacao com RANSAC.
    if (pts_from.size() < 4) {
        return cv::Mat();
    }

    // 5. Estimativa de transformacao de SIMILARIDADE (translacao + rotacao + escala
    //    uniforme, 4 GdL) via RANSAC. Modelo adequado ao focus stacking: a variacao
    //    entre frames e apenas foco -> leve ampliacao + micro deslocamento/rotacao.
    //    Diferente da homografia completa, nao admite perspectiva/cisalhamento, o que
    //    elimina por construcao os warps radiais "explosivos".
    cv::Mat inlier_mask;
    cv::Mat M = cv::estimateAffinePartial2D(pts_from, pts_to, inlier_mask, cv::RANSAC, 3.0);
    if (M.empty()) {
        return cv::Mat();
    }

    // 5.1 Validacao de robustez: um numero minimo de inliers consistentes garante que
    // a transformacao nao foi calculada sobre correspondencias ambiguas (textura repetitiva).
    const int min_inliers = 15;
    const int inliers = cv::countNonZero(inlier_mask);
    if (inliers < min_inliers) {
        spdlog::warn("    [Align] Par rejeitado: inliers insuficientes ({} < {}).", inliers, min_inliers);
        return cv::Mat();
    }

    // 5.2 Rejeita transformacoes com escala implausivel (colapso/explosao).
    if (!isSimilaritySane(M)) {
        spdlog::warn("    [Align] Par rejeitado: similaridade degenerada (escala implausivel).");
        return cv::Mat();
    }

    // 6. Retorna a transformacao homogenea 3x3 (from -> to).
    return toHomogeneous(M);
}

}
