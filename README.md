# Clasificador de Bacterias mediante un Perceptrón Simple

## Descripción

Este proyecto implementa desde cero un perceptrón simple utilizando Python.

El objetivo consiste en entrenar una neurona artificial capaz de clasificar bacterias en dos categorías:

- 🟢 Bacteria benigna (Clase 0)
- 🔴 Bacteria maligna (Clase 1)

El aprendizaje se realiza mediante el algoritmo clásico de corrección del error propuesto por Frank Rosenblatt.

---

## Objetivos

- Implementar un perceptrón simple desde cero.
- Ajustar iterativamente los pesos sinápticos y el bias.
- Validar el modelo utilizando datos no vistos durante el entrenamiento.
- Visualizar la frontera de decisión aprendida.

---

## Dataset

El proyecto utiliza un dataset sintético diseñado específicamente para problemas linealmente separables.

Cada fila representa una bacteria independiente.

| Columna          | Descripción                           |
| ---------------- | ------------------------------------- |
| id               | Identificador de la muestra           |
| tamano_um        | Longitud promedio de la bacteria (μm) |
| crecimiento_hora | Índice de crecimiento                 |
| clase            | 0 = Benigna, 1 = Maligna              |

---

## Medición del tamaño

El tamaño corresponde a la longitud promedio observada mediante microscopía.

La longitud real puede estimarse mediante:

$L=\frac{L_p\cdot FE}{M}$

donde:

- **L** = longitud real.
- **Lp** = longitud medida sobre la imagen.
- **FE** = factor de escala.
- **M** = aumento del microscopio.

Los tamaños utilizados oscilan aproximadamente entre **1 μm y 9.5 μm**.

---

## Índice de crecimiento

El índice de crecimiento representa la capacidad reproductiva de la bacteria bajo condiciones ideales de laboratorio.

Valores bajos indican bacterias de reproducción lenta.

Valores elevados representan bacterias capaces de multiplicarse rápidamente.

---

## Entrenamiento

El dataset se divide automáticamente en:

- **66 %** para entrenamiento.
- **34 %** para validación.

Durante el entrenamiento el perceptrón:

1. Calcula el potencial postsináptico.
2. Aplica una función de activación escalón.
3. Calcula el error.
4. Actualiza los pesos.
5. Repite el proceso hasta que los pesos convergen.

---

## Regla de aprendizaje

Para cada peso:

$w_{nuevo}=w_{viejo}+\eta\cdot error\cdot entrada$

donde:

- η = tasa de aprendizaje.
- error = salida esperada − salida obtenida.

---

## Validación

Finalizado el entrenamiento, el modelo clasifica el conjunto de validación.

Se muestran:

- Predicción de cada bacteria.
- Cantidad de aciertos.
- Cantidad de errores.
- Precisión final.

---

## Visualización

El proyecto genera un gráfico donde:

- Cada punto representa una bacteria.
- Los colores representan la clase.
- La recta corresponde a la frontera de decisión aprendida por el perceptrón.

---

## Tecnologías

- Python
- Pandas
- Matplotlib

---

## Estructura del proyecto

```text
data/
│── bacterias.csv

model/
│── Bacterium.py
│── Perceptron.py

utils/
│── CsvReader.py
│── DatasetSplitter.py

visualization/
│── LinearFunction.py

main.py
```

---
