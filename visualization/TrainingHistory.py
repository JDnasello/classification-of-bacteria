import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator

class TrainingHistory:

    @staticmethod
    def plot(history):
        """
        Función que implementa matplotlib para generar un gráfico de la curva de aprendizaje del perceptrón.
        Muestra la cantidad de errores cometidos en cada época.
        """

        # Eje X: épocas
        eras = range(1, len(history) + 1)

        # Eje Y: errores por época
        plot.figure(figsize=(8, 5))
        plot.plot(eras, history, marker="o", linewidth=2)

        plot.title("Evolución del entrenamiento")
        plot.xlabel("Época")
        plot.ylabel("Errores")
        plot.grid(True)

        plot.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plot.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plot.show()