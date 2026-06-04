#pragma once
#include "align/align_strategy.hpp"

namespace fs_core
{

    /**
     * @brief Implementacao de alinhamento baseada em Enhanced Correlation Coefficient (ECC).
     * * Maximiza o coeficiente de correlacao (fidelidade fotometrica) entre as imagens.
     */
    class ECCAlign : public AlignStrategy
    {
    public:
        /**
         * @brief Construtor com parametros de convergencia.
         * @param iterations Numero maximo de iteracoes do gradiente descendente.
         */
        explicit ECCAlign(int iterations = 500, double eps = 1e-5);
        ~ECCAlign() override = default;

        [[nodiscard]] virtual ImageType align(const ImageType &target, const ImageType &reference, cv::Rect &global_roi);

    private:
        int number_of_iterations_;
        double termination_eps_;
    };

} 