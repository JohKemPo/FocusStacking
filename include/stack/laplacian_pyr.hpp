#pragma once
#include "stack/stack_strategy.hpp"
#include <vector>

namespace fs_core {

/**
 * @brief Empilhamento avancado usando Fusao de Piramide Laplaciana (Multiescala).
 * * Decompoe as imagens em bandas de frequencia e funde cada banda separadamente.
 * Mantem rigorosamente a logica do artigo de Wang and Chang (2011) solicitada
 * no codigo original, porem evitando alocacao global de N imagens na RAM.
 */
class LaplacianPyramidStacker : public StackStrategy {
public:
    explicit LaplacianPyramidStacker(int depth = 5, int kernel_size = 5);
    ~LaplacianPyramidStacker() override = default;

    void addImage(const ImageType& aligned_img) override;
    [[nodiscard]] ImageType finalize() override;

private:
    int depth_;
    int kernel_size_;
    bool is_first_image_;

    std::vector<cv::UMat> fused_lap_pyr_;
    cv::UMat fused_base_;

    cv::UMat max_deviation_map_;
    std::vector<cv::UMat> max_region_energy_;

    void generatePyramid(const ImageType& img, std::vector<cv::UMat>& lap_pyr, cv::UMat& base);
    cv::UMat calculateDeviation(const cv::UMat& src) const;
    cv::UMat calculateRegionEnergy(const cv::UMat& src) const;
};

} 