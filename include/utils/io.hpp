#pragma once
#include "core/types.hpp"
#include <string>
#include <vector>

namespace fs_core {

/**
 * @brief Gestor de I/O isolando o acesso ao disco.
 */
class IOManager {
public:
    /**
     * @brief Retorna uma lista ordenada alfabeticamente de ficheiros de imagem validos.
     */
    static std::vector<std::string> getImagesFromDirectory(const std::string& dir_path);

    /**
     * @brief Verifica se um caminho e um diretorio (util para modo batch).
     */
    static bool isDirectory(const std::string& path);

    /**
     * @brief Retorna os subdiretorios (lotes individuais) dentro de uma pasta mestre.
     */
    static std::vector<std::string> getSubdirectories(const std::string& dir_path);

    /**
     * @brief Abstrai o carregamento (leitura) da imagem para a RAM/VRAM.
     */
    static ImageType loadImage(const std::string& path);

    /**
     * @brief Abstrai a escrita do artefato final.
     */
    static void saveImage(const std::string& path, const ImageType& img);
};

} 