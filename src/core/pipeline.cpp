#include "core/pipeline.hpp"
#include "align/ecc_align.hpp"
#include "align/feature_align.hpp"
#include "stack/naive_log.hpp"
#include "stack/laplacian_pyr.hpp"
#include "utils/io.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/core/ocl.hpp>

namespace fs_core
{

    Pipeline::Pipeline(const Config &config) : config_(config)
    {
        configureHardware();
        initializeAligner();
        initializeStacker();
    }

    void Pipeline::configureHardware()
    {
        if (!config_.use_gpu)
        {
            cv::ocl::setUseOpenCL(false);
            spdlog::info("Aceleracao OpenCL/GPU desativada via configuracao.");
        }
        else if (cv::ocl::haveOpenCL())
        {
            cv::ocl::setUseOpenCL(true);
            spdlog::info("Aceleracao OpenCL ativada. Dispositivo associado: {}", cv::ocl::Device::getDefault().name());
        }
        else
        {
            spdlog::warn("Uso de GPU solicitado, mas OpenCL esta indisponivel. Fallback para rotinas de CPU.");
        }
    }

    void Pipeline::initializeAligner()
    {
        if (config_.align_method == "ecc")
        {
            aligner_ = std::make_unique<ECCAlign>();
        }
        else if (config_.align_method == "kaze")
        {
            aligner_ = std::make_unique<FeatureAlign>(FeatureAlign::FeatureType::KAZE, config_.max_features);
        }
        else if (config_.align_method == "sift")
        {
            aligner_ = std::make_unique<FeatureAlign>(FeatureAlign::FeatureType::SIFT, config_.max_features);
        }
        else if (config_.align_method == "orb")
        {
            aligner_ = std::make_unique<FeatureAlign>(FeatureAlign::FeatureType::ORB, config_.max_features);
        }
        else
        {
            spdlog::warn("Metodo de alinhamento invalido: {}. Aplicando fallback para ECC.", config_.align_method);
            aligner_ = std::make_unique<ECCAlign>();
        }
    }

    void Pipeline::initializeStacker()
    {
        if (config_.stack_method == "naive_log")
        {
            stacker_ = std::make_unique<NaiveLoGStacker>(config_.kernel_size);
        }
        else
        {
            stacker_ = std::make_unique<LaplacianPyramidStacker>(config_.pyr_depth, config_.kernel_size);
        }
    }

    void Pipeline::processDirectory(const std::string &input_dir, const std::string &output_path)
    {
        // A logica de varredura e ordenacao foi delegada ao IOManager
        std::vector<std::string> image_paths = IOManager::getImagesFromDirectory(input_dir);

        if (image_paths.empty())
        {
            spdlog::error("Nenhuma imagem suportada encontrada no diretorio de entrada: {}", input_dir);
            throw PipelineException("Input directory vazio ou sem formatos suportados.");
        }

        spdlog::info("Inicializando rotina com {} imagens.", image_paths.size());

        // Leitura abstraida via IOManager
        ImageType reference = IOManager::loadImage(image_paths[0]);
        if (reference.empty())
            throw PipelineException("Falha ao carregar a imagem de referencia: " + image_paths[0]);

        cv::Rect global_roi(0, 0, reference.cols, reference.rows);

        stacker_->addImage(reference);

        for (size_t i = 1; i < image_paths.size(); ++i)
        {
            spdlog::info("Processando camada {}/{} -> [{}]", i + 1, image_paths.size(), image_paths[i]);

            ImageType current = IOManager::loadImage(image_paths[i]);
            if (current.empty())
                continue;

            ImageType aligned = aligner_->align(current, reference, global_roi);
            stacker_->addImage(aligned);
        }

        spdlog::info("Renderizando imagem consolidada...");
        ImageType final_result = stacker_->finalize();

        if (config_.crop_align && global_roi.area() > 0 && global_roi.area() != final_result.size().area())
        {
            spdlog::info("Aplicando crop de intersecção segura. Novo tamanho: {}x{}", global_roi.width, global_roi.height);

            final_result = final_result(global_roi).clone();
        }

        IOManager::saveImage(output_path, final_result);
        spdlog::info("Artefato final escrito com sucesso no diretorio: {}", output_path);
    }

} 