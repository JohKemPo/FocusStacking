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

