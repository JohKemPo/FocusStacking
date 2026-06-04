#pragma once
#include "core/types.hpp"

namespace fs_core
{

    /**
     * @brief Interface base para estrategias de alinhamento de imagens.
     */
    class AlignStrategy
    {
    public:
        virtual ~AlignStrategy() = default;

        /**
         * @brief Alinha a imagem alvo em relacao a imagem de referencia.
         * * @param target Imagem a ser alinhada.
         * @param reference Imagem de referencia (base fixa).
         * @return ImageType Imagem resultante apos a transformacao geométrica.
         */
        [[nodiscard]] virtual ImageType align(const ImageType &target, const ImageType &reference, cv::Rect &global_roi) = 0;
    };

}