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
         * @brief Estima a transformacao geometrica que leva `from` ao referencial de `to`.
         *
         * Retorna uma matriz 3x3 (CV_64F) em coordenadas homogeneas tal que, para um ponto
         * p em `from`, H*p fornece a posicao correspondente em `to`. O pipeline usa essa
         * matriz para encadear transformacoes entre vizinhos (alinhamento sequencial) e
         * aplicar o warp uma unica vez no referencial global.
         *
         * @param from Imagem cujas coordenadas serao mapeadas.
         * @param to   Imagem de referencia local (vizinho imediato na sequencia).
         * @return cv::Mat Matriz 3x3 (from -> to), ou matriz vazia se o alinhamento falhar.
         */
        [[nodiscard]] virtual cv::Mat estimateTransform(const ImageType &from, const ImageType &to) = 0;
    };

}