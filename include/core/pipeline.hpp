#pragma once
#include "core/types.hpp"
#include "align/align_strategy.hpp"
#include "stack/stack_strategy.hpp"
#include <memory>
#include <string>

namespace fs_core {

/**
 * @brief Gerenciador principal do fluxo de trabalho de Focus Stacking.
 * * Orquestra a leitura incremental de diretórios (Streaming/Batch), 
 */
class Pipeline {
public:
    /**
     * @brief Construtor do Pipeline recebendo a configuração central.
     * @param config Estrutura com parâmetros lidos do YAML/CLI.
     */
    explicit Pipeline(const Config& config);

    // Desabilitar cópias para evitar duplicação não intencional de instâncias de processamento pesado
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * @brief Executa o pipeline de empilhamento em um diretório específico.
     * * @param input_dir Diretório contendo a sequência de imagens (ex: img001.tif, img002.tif).
     * @param output_path Caminho final do arquivo gerado (ex: ./result.tiff).
     */
    void processDirectory(const std::string& input_dir, const std::string& output_path);

private:
    Config config_;
    
    // Instâncias concretas abstraídas através do padrão Strategy
    std::unique_ptr<AlignStrategy> aligner_;
    std::unique_ptr<StackStrategy> stacker_;

    // Factory methods internos para injeção da lógica correta baseada no config_
    void initializeAligner();
    void initializeStacker();
    
    // Função auxiliar para inicializar a aceleração por hardware
    void configureHardware();

    void processSingleStack(const std::vector<std::string>& image_paths, const std::string& output_path);
};

} 