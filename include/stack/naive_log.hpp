#pragma once
#include "stack/stack_strategy.hpp"

namespace fs_core {

/**
 * @brief Empilhamento usando o maximo do Laplaciano do Gaussiano (Pixel-wise).
 * * Metodo basico: aplica blur, calcula o Laplaciano e retem o pixel que 
 */
class NaiveLoGStacker : public StackStrategy {
public:
    explicit NaiveLoGStacker(int kernel_size = 3);
    ~NaiveLoGStacker() override = default;

    void addImage(const ImageType& aligned_img) override;
    [[nodiscard]] ImageType finalize() override;

private:
    int kernel_size_;
    bool is_first_image_;
    
    ImageType canvas_;
    cv::UMat max_log_map_;
};

} 