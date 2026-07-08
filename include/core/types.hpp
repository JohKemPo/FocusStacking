#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fs_core {

using ImageType = cv::UMat;

struct Config {
    std::string align_method = "ecc";        // "ecc", "kaze", "sift", "orb", "none"
    std::string stack_method = "laplacian";  // "laplacian", "naive_log"
    int pyr_depth = 5;                       // Profundidade da pirâmide Laplaciana
    int kernel_size = 5;                     // Tamanho do kernel gaussiano
    bool use_gpu = true;                     // Flag para habilitar/desabilitar OpenCL
    std::string output_format = ".tiff";     // ".tiff", ".png", ".jpg"
    int max_features = 1000;                 // Opcional para métodos de features
    bool crop_align = true;
    
    std::string reference_strategy = "middle"; // "first", "middle", "last"
    int images_per_stack = 15;                 // Quantidade de fotos por empilhamento
};

class PipelineException : public std::runtime_error {
public:
    explicit PipelineException(const std::string& message) : std::runtime_error(message) {}
};

}