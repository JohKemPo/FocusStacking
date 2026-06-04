#pragma once
#include <string>

namespace fs_core {

/**
 * @brief Configura o sistema central de logs da aplicacao.
 * Envia mensagens simultaneamente para a consola (stdout) e para um ficheiro fisico.
 */
class Logger {
public:
    static void init(const std::string& log_file_path = "focus_stacking.log");
};

} 