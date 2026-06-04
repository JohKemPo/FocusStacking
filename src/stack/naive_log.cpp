#include "stack/naive_log.hpp"
#include <opencv2/imgproc.hpp>

namespace fs_core {

NaiveLoGStacker::NaiveLoGStacker(int kernel_size) 
    : kernel_size_(kernel_size), is_first_image_(true) {}

void NaiveLoGStacker::addImage(const ImageType& aligned_img) {
    cv::UMat gray, blurred, laplacian, abs_laplacian;

    // 1. Conversao para tons de cinza para avaliar nitidez de forma isotropica
    if (aligned_img.channels() == 3) {
        cv::cvtColor(aligned_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = aligned_img;
    }

    // 2. Aplicar Gaussian Blur para reduzir ruido (evitar que ruido dispare o foco)
    cv::GaussianBlur(gray, blurred, cv::Size(kernel_size_, kernel_size_), 3.0);

    // 3. Laplaciano (CV_64F interno para evitar overflow direcional)
    cv::Laplacian(blurred, laplacian, CV_64F, kernel_size_);
    
    // 4. Valor absoluto representa a magnitude da borda
    laplacian.convertTo(abs_laplacian, CV_32F);
    cv::absdiff(abs_laplacian, cv::Scalar::all(0), abs_laplacian);

    if (is_first_image_) {
        aligned_img.copyTo(canvas_);
        max_log_map_ = abs_laplacian;
        is_first_image_ = false;
        return;
    }

    // 5. Comparacao Streaming: Criar mascara onde a imagem ATUAL e mais nitida
    cv::UMat mask;
    cv::compare(abs_laplacian, max_log_map_, mask, cv::CMP_GT);

    // 6. Atualizacao in-place das matrizes
    aligned_img.copyTo(canvas_, mask);
    abs_laplacian.copyTo(max_log_map_, mask);
}

ImageType NaiveLoGStacker::finalize() {
    return canvas_;
}

} 