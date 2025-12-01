# Deep Learning como Problema de Control Óptimo

## Creando el entorno y ejecutando los informes

Descargar e instalar [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) si no se tiene instalado.

Descargar el repositorio:
```bash
git clone https://github.com/...git
cd DeepLearningOptimalControl
```

Crear el entorno de conda e instalar las dependencias:
```bash
conda env create -f environment.yml
conda activate DeepOptimalControl
```

## Introducción y resumen

La clasificación supervisada es una tarea central del aprendizaje de máquinas: dado un conjunto etiquetado $\{(x_i,c_i)\}_{i=1}^m$, con $x_i\in\mathbb{R}^n$ y $c_i\in\{c_0,c_1\}$, buscamos un clasificador $g:\mathbb{R}^n\to\{c_0,c_1\}$ que asigne correctamente la clase a nuevas observaciones. Un enfoque estándar consiste en entrenar un modelo paramétrico minimizando una pérdida de ajuste (por ejemplo, mínimos cuadrados) sobre los datos.

En este proyecto se estudia una interpretación estructural: una red residual (ResNet) puede verse como la discretización temporal (Forward Euler) de un sistema dinámico, donde la *profundidad* juega el rol de *tiempo* y los pesos/sesgos actúan como *controles*. Es decir, para cada muestra,
$$
y_i^{[0]}=x_i,\qquad
y_i^{[j+1]} = y_i^{[j]} + \Delta t\, \sigma\!\big(K^{[j]}y_i^{[j]}+\beta^{[j]}\big),\quad j=0,\dots,N-1,
$$
y el clasificador se construye a partir de la salida $y_i^{[N]}$ mediante un lector lineal $(W,\mu)$ y una
hipótesis fija $C$.

**Problema.** Formular el entrenamiento de la ResNet como un problema de control óptimo (tipo Mayer): encontrar parámetros $u=\{(K^{[j]},\beta^{[j]})\}_{j=0}^{N-1}$ (y eventualmente $W,\mu$) que minimicen
$$
J(u,W,\mu)=\frac12\sum_{i=1}^m \Big|\, C\!\big(W\,y_i^{[N]}+\mu\big)-c_i \Big|^2
\quad \text{s.a. la dinámica discreta anterior.}
$$
A partir de esta formulación, el objetivo es derivar condiciones óptimas (p.ej. PMP/HJB), proponer e implementar un algoritmo numérico basado en ecuaciones estado--adjunta (forward/backward) y comparar desempeño y complejidad computacional frente a clasificadores habituales (regresión, árboles, etc.). 
