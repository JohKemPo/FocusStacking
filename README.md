
# FocusStackingCPP

Este projeto é uma implementação de alto desempenho em C++20 para o empilhamento de foco (Focus Stacking) multiescala, desenhado primariamente para microscopia digital e processamento científico de imagens.

O sistema substitui abordagens tradicionais baseadas em memória estática por uma arquitetura orientada a fluxo (streaming) combinada com aceleração de hardware assíncrona, viabilizando o processamento de lotes massivos de imagens de altíssima resolução (4K, 8K) sem exaustão de memória RAM.

## 1. Visão Geral e Modelo Mental

Em abordagens ingénuas ou implementadas em linguagens de tipagem dinâmica (como a versão anterior em Python), as matrizes multidimensionais de todas as imagens da pilha geométrica são carregadas integralmente na memória (complexidade espacial $O(N)$). Na fusão por Pirâmide Laplaciana, isto exige armazenar múltiplas escalas para cada camada temporal, resultando num rápido *thrashing* do sistema operativo e *crashes* por falta de RAM.

**A Solução Arquitetural:** Este sistema processa as imagens sequencialmente através de um *Pipeline Acumulador*. Apenas duas imagens residem na memória simultaneamente (a camada atual e o estado consolidado). A complexidade espacial passa de $O(N)$ para $O(1)$ em relação à profundidade da pilha (Z-stack).

## 2. Trade-offs Arquiteturais e Decisões de Engenharia

Durante o projeto, as seguintes trocas (trade-offs) foram assumidas para priorizar o desempenho científico:

* **Precisão vs. Velocidade no Alinhamento:** O alinhamento tradicional baseado em features (KAZE/SIFT) é rápido, mas frequentemente falha em microscopia biológica devido à ausência de arestas duras (high frequency edges) em camadas fora de foco. Optou-se pela introdução do **ECC (Enhanced Correlation Coefficient)**. Ele otimiza a variância fotométrica (alinhamento subpixel) ao custo de maior consumo de CPU iterativa, mas garante uma fidelidade científica superior. Fallbacks para KAZE/SIFT foram mantidos para macrofotografia genérica.
* **CPU vs. Tráfego PCIe (GPU):** A estrutura utiliza `cv::UMat` (API Transparente do OpenCV). As operações morfológicas, convoluções (Filtros Gaussianos/Laplacianos) e cálculos algébricos ocorrem nativamente via OpenCL (ou CUDA). O trade-off é um ligeiro *overhead* no envio inicial da matriz pela barra PCIe, largamente compensado pela paralelização massiva na GPU, reduzindo o aquecimento térmico do processador central.
* **Variância Local vs. Entropia:** O artigo original de Wang e Chang (2011) calcula a energia e entropia da base da pirâmide via histogramas. Construir histogramas móveis em janelas deslizantes via CPU arruinaria a performance em imagens 4K. Matematicamente, a variância capta a mesma informação de alta frequência (nitidez e contraste estrutural). Aplicámos o teorema matemático $Var(X) = E[X^2] - (E[X])^2$ através de filtros de caixa (`cv::boxFilter`) vetorizados, alcançando o mesmo objetivo heurístico de forma 100x mais rápida.

## 3. Pré-requisitos e Dependências

O projeto utiliza bibliotecas modernas em C++ e deve ser gerido preferencialmente via `vcpkg`.

* Compilador com suporte a **C++20** (GCC 11+, Clang 13+ ou MSVC 19.30+).
* **CMake** >= 3.20
* **Dependências (`vcpkg`):**
  * `opencv4` (Compilado idealmente com módulos: `core`, `imgproc`, `video`, `calib3d`, `features2d` e suporte OpenCL/TBB).
  * `spdlog` (Para logs estruturados e auditoria científica).
  * `yaml-cpp` (Para gestão de configurações em lote).

## 4. Instruções de Compilação (Linux/WSL)

1. Instale o `vcpkg` (caso não o possua):
   ```bash
   git clone [https://github.com/microsoft/vcpkg.git](https://github.com/microsoft/vcpkg.git)
   ./vcpkg/bootstrap-vcpkg.sh
   ./vcpkg/vcpkg install opencv4[core,imgproc,video,calib3d,features2d] spdlog yaml-cpp
    ```

2. Configure o build utilizando a toolchain do vcpkg:

    ```bash
    mkdir build && cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
    ```
3. Compile o executável otimizado:

    ```bash
    make -j$(nproc)
    ```

## 5. Estrutura de Configuração (YAML)

O sistema deve ser ajustado sem necessidade de recompilação através de um ficheiro `.yaml`.
Exemplo: `config/default.yaml`

    ```bash
    align_method: "ecc"            # Opções: ecc (recomendado microscopia), kaze, sift, orb, none
    stack_method: "laplacian"      # Opções: laplacian (Multiescala), naive_log (Fallback ultra-rápido)
    pyr_depth: 6                   # Profundidade da pirâmide (5-7 é o ideal para resoluções >1080p)
    kernel_size: 5                 # Tamanho da matriz Gaussiana (Obrigatório ser número ímpar)
    use_gpu: true                  # Envia operações pesadas para VRAM via OpenCL
    output_format: ".tiff"         # Tiff preserva o range dinâmico sem perdas
    max_features: 1500             # Limite de features caso use KAZE/SIFT/ORB
    ```

## 6. Utilização (CLI)

A aplicação é executada via terminal de comandos.

Processamento simples de um diretório:

    ```bash
    ./FocusStackingCPP --input /caminho/para/pasta/imagens --output ./resultado_final.tiff
    ```

Processamento com hiperparâmetros customizados (Modo Batch/Científico):

    ```bash
    ./FocusStackingCPP --input /caminho/para/pasta/imagens --output ./resultado_final.tiff --config ../config/default.yaml
    ```


Nota: As imagens na pasta de entrada devem ser suportadas pelo OpenCV (.tif, .jpg, .png) e serão ordenadas alfabeticamente. A primeira imagem é sempre adotada como âncora de referência geométrica e fotométrica do alinhamento espacial.

## 7. Logs e Diagnósticos

Uma execução bem-sucedida imprimirá na consola (e gravará no ficheiro de auditoria `focus_stacking_session.log`) o estado de ativação do hardware, a estratégia instanciada, o descarte de outliers e a latência de I/O em disco. Utilize este registo de log para perfilar gargalos caso o seu disco não consiga acompanhar o rendimento da GPU.



<br>
<br>
<br>
<br>

-------------------------------------


# FocusStacking


```
mkdir -p ./{config,include/{core,align,stack,utils},src/{core,align,stack},tests} && \
touch ./CMakeLists.txt \
./config/default.yaml \
./include/core/pipeline.hpp \
./include/core/types.hpp \
./include/align/align_strategy.hpp \
./include/align/ecc_align.hpp \
./include/align/feature_align.hpp \
./include/stack/stack_strategy.hpp \
./include/stack/naive_log.hpp \
./include/stack/laplacian_pyr.hpp \
./include/utils/io.hpp \
./include/utils/logger.hpp \
./src/core/pipeline.cpp \
./src/align/ecc_align.cpp \
./src/align/feature_align.cpp \
./src/stack/naive_log.cpp \
./src/stack/laplacian_pyr.cpp \
./src/main.cpp
```


No seu pipeline de Focus Stacking (`align_method`), existem dois paradigmas matemáticos completamente distintos para alinhar as imagens. Para entender qual usar no seu problema de *focus breathing* (o "zoom" variável), é essencial entender o modelo mental por trás de cada um, seus limites geométricos e os trade-offs computacionais.

Aqui está o funcionamento interno e os fundamentos de cada método suportado pelo seu código.


```
rm CMakeCache.txt 
cmake ..                                                                                                                                                          py base at 08:07:59
make -j$(nproc)
./FocusStackingCPP --input /home/joh/Documentos/FocusStacking/pilha/Ex2/JPG --output ../results/resultado_macro_sift_2_1.tiff --config ../config/default.yaml   
```



---

### 1. O Paradigma de Intensidade (Dense Alignment)

#### **ECC (Enhanced Correlation Coefficient)**

* **O Modelo Mental:** Imagine pegar duas transparências (fotografias), colocar uma sobre a outra contra uma luz forte e movê-las aos poucos. Você tenta encontrar o encaixe perfeito medindo quanta luz passa. O algoritmo não procura objetos; ele olha para **todos os pixels ao mesmo tempo** e tenta minimizar a diferença matemática (erro fotométrico) entre eles.
* **O Funcionamento Interno:** Ele resolve um problema de otimização iterativa. Começa assumindo que a imagem está alinhada e vai derivando a diferença de pixels (gradiente espacial) para saber para qual lado deve "empurrar" a matriz de transformação a cada iteração (usando o método de Newton-Raphson modificado).
* **Trade-offs:**
* **Vantagens:** Altíssima precisão submétrica (sub-pixel). Se as imagens já estiverem muito parecidas, ele faz um ajuste fino perfeito. Lida bem com pouca textura (onde os outros algoritmos falham).
* **Desvantagens:** É míope. Se a imagem inicial estiver muito deslocada, distorcida ou, no seu caso, **com uma diferença de escala muito grande**, a otimização não converge (diverge e a imagem "derrete"). O ECC também sofre ao tentar correlacionar texturas nítidas com texturas borradas (o que acontece em todo *z-stack*).



---

### 2. O Paradigma de Features (Sparse Alignment)

Este paradigma funciona em três etapas fundamentais:

1. **Deteção:** Encontrar "cantos" ou pontos de interesse na imagem que não se confundam com os vizinhos.
2. **Descrição:** Extrair a "impressão digital" matemática do entorno desse ponto.
3. **Matching:** Ligar a impressão digital da imagem A com a imagem B.

Aqui entram as diferenças de SIFT, KAZE e ORB.

#### **SIFT (Scale-Invariant Feature Transform)**

* **O Modelo Mental:** Pense em tentar reconhecer uma constelação no céu noturno. Não importa se você está usando um telescópio com mais zoom ou olhando a olho nu (escala), nem se você tombou a cabeça (rotação) — a proporção geométrica e as distâncias relativas entre as estrelas continuam formando a mesma constelação.
* **O Funcionamento Interno:** Para garantir que a feature seja encontrada independentemente do nível de "zoom", o SIFT aplica um Desfoque Gaussiano iterativo na imagem (criando o *Scale Space* ou Pirâmide Gaussiana). Ele então subtrai imagens de desfoques diferentes (Difference of Gaussians - DoG) para encontrar pontos que sobrevivem a essa agressão visual. O descritor (a impressão digital) é um histograma da direção das bordas ao redor daquele ponto num vetor de 128 dimensões.
* **Trade-offs:**
* **Vantagens:** É o **padrão ouro** absoluto para lidar com variação de escala (*focus breathing*). O SIFT foi construído matematicamente para não se importar com o tamanho aparente do objeto.
* **Desvantagens:** Pesado em CPU e consome bastante RAM, por envolver o cálculo de derivadas múltiplas em ponto flutuante.



#### **KAZE**

* **O Modelo Mental:** Imagine tentar simplificar a pintura de um quadro para focar nos contornos reais. O SIFT usa um embaçamento cego (Gaussiano) que borra tudo por igual. O KAZE pensa: *"Vou borrar apenas as texturas internas (como a pele de um rosto), mas vou me recusar a borrar onde existe uma borda forte (o limite do rosto contra o fundo)"*.
* **O Funcionamento Interno:** O KAZE cria o espaço de escala não usando um borrão simples, mas equações diferenciais de **difusão não-linear** (esquemas baseados em Perona-Malik). O calor matemático flui pela imagem, mas "bate e volta" quando encontra uma aresta dura (um gradiente muito íngreme).
* **Trade-offs:**
* **Vantagens:** Os pontos que ele encontra costumam fazer mais sentido anatômico em macrofotografia e ciência de materiais do que os do SIFT. É extremamente robusto a ruído no sensor.
* **Desvantagens:** Computacionalmente é o mais lento de todos. A resolução numérica de equações de difusão por toda a imagem cobra um preço alto em performance.



#### **ORB (Oriented FAST and Rotated BRIEF)**

* **O Modelo Mental:** Em vez de analisar o gradiente das bordas (SIFT) ou o fluxo de difusão (KAZE), o ORB funciona como um "leitor dinâmico". Ele olha o pixel central, pega um círculo ao redor, e simplesmente testa de forma binária: *"Este pixel no círculo é mais claro ou mais escuro que o centro?"*. Ele anota isso como `1` ou `0` criando uma string rápida (`01011100...`).
* **O Funcionamento Interno:** Foi projetado por pesquisadores da OpenCV para ser uma alternativa ultrarrápida, baseada em testes binários que os processadores resolvem em um único ciclo de clock via operações *bitwise* (XOR e POPCNT).
* **Trade-offs:**
* **Vantagens:** Desempenho absurdo (em tempo real). Usado muito para navegação de drones e robótica.
* **Desvantagens:** Ele é invariante à rotação, mas tem **péssima invariância à escala**. Se o objeto na foto de foco inicial estiver muito menor do que na foto final, os testes de intensidade ao redor do pixel não batem, os descritores mudam, o matching falha, e o *focus stacking* não consegue amarrar a imagem.



---

### Diagnóstico para o seu cenário

Para o problema reportado (você tem imagens de um objeto parado com a câmera fixa, porém o ajuste focal causou variação de "zoom"):

1. O **ORB** vai falhar por sua vulnerabilidade matemática à escala.
2. O **ECC** vai divergir porque ele se baseia num fluxo contínuo e se perde quando tenta alinhar algo nítido com algo que sofreu o borrão óptico da variação do foco.
3. A escolha técnica correta para o seu `config/default.yaml` deve ser o **`sift`**. Ele foi matematicamente projetado para extrair correlações mesmo quando a lente deforma a percepção de profundidade, tamanho e nitidez. Aliado à trava de 4 Graus de Liberdade (`estimateAffinePartial2D`) aplicada anteriormente, ele forçará todas as camadas para o exato mesmo tamanho em que a foto da base se encontrava.

