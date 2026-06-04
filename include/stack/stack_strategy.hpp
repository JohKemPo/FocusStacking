#pragma once
#include "core/types.hpp"

namespace fs_core {

/**
 * @brief Interface base para as estrategias de empilhamento (Focus Stacking).
 * * Adota um modelo de processamento em fluxo (streaming). Em vez de carregar
 */
class StackStrategy {
public:
    virtual ~StackStrategy() = default;

    /**
     * @brief Adiciona uma imagem alinhada ao acumulador de foco.
     * @param aligned_img Imagem pre-processada pela AlignStrategy.
     */
    virtual void addImage(const ImageType& aligned_img) = 0;

    /**
     * @brief Finaliza o processamento e retorna a imagem empilhada.
     * @return ImageType A imagem final combinada.
     */
    [[nodiscard]] virtual ImageType finalize() = 0;
};

} 