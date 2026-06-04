#pragma once
#include "align/align_strategy.hpp"
#include <opencv2/features2d.hpp>

namespace fs_core
{

    /**
     * @brief Implementacao de alinhamento baseada em deteccao de pontos de interesse.
     * * Extrai keypoints e descritores, realiza o matching (KNN com Ratio Test de Lowe)
     * e calcula a matriz de homografia usando RANSAC.
     */
    class FeatureAlign : public AlignStrategy
    {
    public:
        enum class FeatureType
        {
            KAZE,
            SIFT,
            ORB
        };

        /**
         * @brief Construtor da estrategia de features.
         * @param type Algoritmo de deteccao a ser utilizado.
         * @param max_features Numero maximo de features retidas para evitar overhead computacional.
         */
        explicit FeatureAlign(FeatureType type = FeatureType::KAZE, int max_features = 1000);
        ~FeatureAlign() override = default;

        [[nodiscard]] virtual ImageType align(const ImageType &target, const ImageType &reference, cv::Rect &global_roi);

    private:
        FeatureType type_;
        int max_features_;

        [[nodiscard]] cv::Ptr<cv::Feature2D> createDetector() const;
    };

} 