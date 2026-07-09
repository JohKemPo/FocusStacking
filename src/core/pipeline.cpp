#include "core/pipeline.hpp"
#include "align/ecc_align.hpp"
#include "align/feature_align.hpp"
#include "stack/naive_log.hpp"
#include "stack/laplacian_pyr.hpp"
#include "utils/io.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/core/ocl.hpp>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

namespace
{

    // Projeta os cantos de uma imagem `img_size` pela transformacao homogenea `H` e
    // intersecta o "inner bounding box" resultante no ROI global (guilhotina de recorte).
    void updateGlobalRoi(cv::Rect &global_roi, const cv::Mat &H,
                         const cv::Size &img_size, const cv::Size &ref_size)
    {
        std::vector<cv::Point2f> corners = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(img_size.width), 0),
            cv::Point2f(static_cast<float>(img_size.width), static_cast<float>(img_size.height)),
            cv::Point2f(0, static_cast<float>(img_size.height))};

        std::vector<cv::Point2f> tc;
        cv::perspectiveTransform(corners, tc, H);

        int start_x = std::max({0, static_cast<int>(tc[0].x), static_cast<int>(tc[3].x)});
        int start_y = std::max({0, static_cast<int>(tc[0].y), static_cast<int>(tc[1].y)});
        int end_x = std::min({ref_size.width, static_cast<int>(tc[1].x), static_cast<int>(tc[2].x)});
        int end_y = std::min({ref_size.height, static_cast<int>(tc[2].y), static_cast<int>(tc[3].y)});

        cv::Rect safe_rect(start_x, start_y, std::max(0, end_x - start_x), std::max(0, end_y - start_y));
        global_roi &= safe_rect;
    }

} // namespace

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

    void Pipeline::processSingleStack(const std::vector<std::string> &image_paths, const std::string &output_path)
    {
        // Limpeza e reinicialização completa da memória do stacker (Evita vazamento de contexto)
        initializeStacker();

        if (image_paths.empty()) return;

        size_t ref_idx = 0;
        if (config_.reference_strategy == "middle") {
            ref_idx = image_paths.size() / 2;
        } else if (config_.reference_strategy == "last") {
            ref_idx = image_paths.size() - 1;
        }

        ImageType reference = IOManager::loadImage(image_paths[ref_idx]);
        if (reference.empty()) {
            spdlog::error("Falha crítica ao carregar a imagem de referência: {}", image_paths[ref_idx]);
            return;
        }

        const cv::Size ref_size = reference.size();
        cv::Rect global_roi(0, 0, ref_size.width, ref_size.height);

        // A referencia entra sem warp (transformacao identidade).
        stacker_->addImage(reference);
        spdlog::info("  Camada de referencia {}/{} -> [{}]", ref_idx + 1, image_paths.size(), image_paths[ref_idx]);

        // Alinhamento SEQUENCIAL: cada frame e alinhado ao seu vizinho imediato (onde a
        // diferenca de foco/breathing e minima e ha muitos matches). As transformacoes
        // vizinho-a-vizinho sao ACUMULADAS ate o referencial global, corrigindo a escala
        // progressiva que causava o smear radial nos frames de foco extremo.
        auto process_chain = [&](int start, int stop, int step)
        {
            cv::Mat cumulative = cv::Mat::eye(3, 3, CV_64F); // vizinho -> referencia
            ImageType neighbor = reference;

            for (int i = start; i != stop; i += step)
            {
                spdlog::info("  Processando camada {}/{} -> [{}]", i + 1, image_paths.size(), image_paths[i]);

                ImageType current = IOManager::loadImage(image_paths[i]);
                if (current.empty()) continue;

                // Transformacao do frame atual para o referencial do vizinho (from -> to).
                cv::Mat T = aligner_->estimateTransform(current, neighbor);
                if (T.empty())
                {
                    // Falha no par: assume ausencia de movimento (identidade) e segue a cadeia.
                    spdlog::warn("    [Align] Par sem alinhamento confiavel; usando identidade neste passo.");
                    T = cv::Mat::eye(3, 3, CV_64F);
                }

                // Compoe: (vizinho -> ref) * (atual -> vizinho) = (atual -> ref).
                cumulative = cumulative * T;

                ImageType aligned;
                cv::warpPerspective(current, aligned, cumulative, ref_size,
                                    cv::INTER_LINEAR, cv::BORDER_REPLICATE);

                updateGlobalRoi(global_roi, cumulative, current.size(), ref_size);
                stacker_->addImage(aligned);

                neighbor = current; // o frame atual vira referencia local do proximo passo
            }
        };

        // Caminha para os dois lados a partir da referencia, encadeando os vizinhos.
        process_chain(static_cast<int>(ref_idx) - 1, -1, -1);                          // ref-1 .. 0
        process_chain(static_cast<int>(ref_idx) + 1, static_cast<int>(image_paths.size()), 1); // ref+1 .. fim

        ImageType final_result = stacker_->finalize();

        if (config_.crop_align && global_roi.area() > 0 && global_roi.area() != final_result.size().area())
        {
            final_result = final_result(global_roi).clone();
        }

        IOManager::saveImage(output_path, final_result);
    }

    void Pipeline::processDirectory(const std::string &input_dir, const std::string &output_path)
    {
        fs::path out_p(output_path);
        fs::path out_dir = out_p.has_extension() ? out_p.parent_path() : out_p;
        std::string base_stem = out_p.has_extension() ? out_p.stem().string() : "resultado";
        std::string ext = out_p.has_extension() ? out_p.extension().string() : config_.output_format;

        if (!out_dir.empty()) {
            fs::create_directories(out_dir);
        }

        // Tenta carregar subdiretórios estruturados (Cenário 1)
        std::vector<std::string> subdirs = IOManager::getSubdirectories(input_dir);

        if (!subdirs.empty()) {
            spdlog::info("Modo Batch Estruturado detectado. Processando {} subpastas.", subdirs.size());
            for (size_t s = 0; s < subdirs.size(); ++s) {
                std::vector<std::string> images = IOManager::getImagesFromDirectory(subdirs[s]);
                if (images.empty()) continue;

                fs::path sub_p(subdirs[s]);
                fs::path current_out = out_dir / (base_stem + "_" + sub_p.filename().string() + ext);

                spdlog::info("[Stack {}/{}] Processando lote da pasta: {}", s + 1, subdirs.size(), sub_p.filename().string());
                processSingleStack(images, current_out.string());
            }
        }
        else {
            // Cenário 2: Pasta linear única contendo todas as 1080 imagens misturadas
            std::vector<std::string> all_images = IOManager::getImagesFromDirectory(input_dir);
            if (all_images.empty()) {
                throw PipelineException("O diretório informado está vazio ou não contém imagens suportadas.");
            }

            size_t chunk_size = static_cast<size_t>(config_.images_per_stack);
            size_t total_images = all_images.size();
            size_t total_stacks = (total_images + chunk_size - 1) / chunk_size;

            spdlog::info("Modo Flat Linear ativo. {} imagens segmentadas em blocos de {} (Total: {} fotos finais).", total_images, chunk_size, total_stacks);

            for (size_t k = 0; k < total_stacks; ++k) {
                size_t start_idx = k * chunk_size;
                size_t end_idx = std::min(start_idx + chunk_size, total_images);

                std::vector<std::string> chunk_images(all_images.begin() + start_idx, all_images.begin() + end_idx);

                std::stringstream ss;
                ss << base_stem << "_" << std::setw(3) << std::setfill('0') << (k + 1) << ext;
                fs::path current_out = out_dir / ss.str();

                spdlog::info("[Stack {}/{}] Processando sequência linear (Índices: {} a {})", k + 1, total_stacks, start_idx, end_idx - 1);
                processSingleStack(chunk_images, current_out.string());
            }
        }
        spdlog::info("Varredura e processamento do dataset de produção finalizado.");
    }

} 