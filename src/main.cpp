#include "core/pipeline.hpp"
#include <iostream>
#include <string>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include "utils/logger.hpp"

using namespace fs_core;

Config parseConfigYAML(const std::string& yaml_path) {
    Config config;
    try {
        YAML::Node node = YAML::LoadFile(yaml_path);
        
        if (node["align_method"]) config.align_method = node["align_method"].as<std::string>();
        if (node["stack_method"]) config.stack_method = node["stack_method"].as<std::string>();
        if (node["pyr_depth"]) config.pyr_depth = node["pyr_depth"].as<int>();
        if (node["kernel_size"]) config.kernel_size = node["kernel_size"].as<int>();
        if (node["use_gpu"]) config.use_gpu = node["use_gpu"].as<bool>();
        if (node["output_format"]) config.output_format = node["output_format"].as<std::string>();
        if (node["max_features"]) config.max_features = node["max_features"].as<int>();
        if (node["crop_align"]) config.crop_align = node["crop_align"].as<bool>();

        if (node["reference_strategy"]) config.reference_strategy = node["reference_strategy"].as<std::string>();
        if (node["images_per_stack"]) config.images_per_stack = node["images_per_stack"].as<int>();
    } catch (const std::exception& e) {
        spdlog::warn("Erro ao interpretar arquivo YAML: {}. Os parametros padrao hardcoded serao aplicados.", e.what());
    }
    return config;
}

void printHelp(const char* binary_name) {
    std::cout << "FocusStacking v1.0 - C++ Scientific Processor\n"
              << "Uso basico: " << binary_name << " --input <pasta_imagens> [--output <arquivo_saida>] [--config <arquivo.yaml>]\n\n"
              << "Parametros:\n"
              << "  --input    [Obrigatorio] Caminho para a pasta contendo a sequencia de imagens (Z-Stack).\n"
              << "  --output   [Opcional] Caminho final de escrita. Default: ./result.tiff\n"
              << "  --config   [Opcional] Caminho para o YAML contendo os hiperparametros de processamento.\n"
              << "  -h, --help Exibe este menu.\n";
}

int main(int argc, char** argv) {
    std::string input_dir = "";
    std::string output_file = "./result.tiff";
    std::string config_file = "";

    try {
        fs_core::Logger::init("focus_stacking_session.log");
    } catch (const std::exception& e) {
        std::cerr << "Falha critica no Logger: " << e.what() << "\n";
        return -1;
    }


    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        }
    }

    if (input_dir.empty()) {
        spdlog::critical("Entrada invalida. A flag --input e mandatória.");
        printHelp(argv[0]);
        return -1;
    }

    Config config;
    if (!config_file.empty()) {
        spdlog::info("Injetando hiperparametros via documento YAML: {}", config_file);
        config = parseConfigYAML(config_file);
    }

    try {
        Pipeline pipeline(config);
        pipeline.processDirectory(input_dir, output_file);
    } catch (const PipelineException& e) {
        spdlog::critical("Processamento abortado. Erro de Pipeline: {}", e.what());
        return -1;
    } catch (const std::exception& e) {
        spdlog::critical("Processamento abortado. Excecao nao tratada na memoria: {}", e.what());
        return -1;
    }

    return 0;
}