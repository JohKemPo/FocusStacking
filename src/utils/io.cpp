#include "utils/io.hpp"
#include <filesystem>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace fs_core {

std::vector<std::string> IOManager::getImagesFromDirectory(const std::string& dir_path) {
    std::vector<std::string> image_paths;
    
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        spdlog::error("Diretorio invalido ou inexistente: {}", dir_path);
        return image_paths;
    }

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".tif" || ext == ".tiff" || ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }

    // Ordenacao obrigatoria para reprodutibilidade
    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

bool IOManager::isDirectory(const std::string& path) {
    return fs::is_directory(path);
}

std::vector<std::string> IOManager::getSubdirectories(const std::string& dir_path) {
    std::vector<std::string> subdirs;
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) return subdirs;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_directory()) {
            subdirs.push_back(entry.path().string());
        }
    }
    std::sort(subdirs.begin(), subdirs.end());
    return subdirs;
}

ImageType IOManager::loadImage(const std::string& path) {
    return cv::imread(path, cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
}

void IOManager::saveImage(const std::string& path, const ImageType& img) {
    if (!cv::imwrite(path, img)) {
        spdlog::error("Falha ao gravar o ficheiro no disco: {}", path);
        throw PipelineException("Erro de I/O na escrita da imagem final.");
    }
}

} 