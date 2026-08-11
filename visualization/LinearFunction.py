import matplotlib.pyplot as plot;

class LinearFunction:

    @staticmethod
    def linear_func(samples, perceptron):

        benign_x = []
        benign_y = []
        harmful_x = []
        harmful_y = []

        benign_count = 0
        harmful_count = 0

        for bacterium in samples:

            classification = perceptron.predict(bacterium.features)
            
            if classification == 0:
                benign_x.append(bacterium.features[0])
                benign_y.append(bacterium.features[1])
                benign_count += 1
            else:
                harmful_x.append(bacterium.features[0])
                harmful_y.append(bacterium.features[1])
                harmful_count += 1

        plot.scatter(benign_x, benign_y, color="green" ,label=f"Benigna ({benign_count})")
        plot.scatter(harmful_x, harmful_y, color="red" ,label=f"Maligna ({harmful_count})")

        w1 = perceptron.weights[0]
        w2 = perceptron.weights[1]
        b = perceptron.bias

        # Se definen 2 valores con el valor mínimo y máximo para x ya que una recta se define con 2 puntos
        x_values = [min(bacterium.features[0] for bacterium in samples), max(bacterium.features[0] for bacterium in samples)]

        # Frontera de decisión
        x2 = [
            - (x * w1 + b)/w2
            for x in x_values
            ]

        plot.title("Clasificador de bacterias")
        plot.grid(True)
        plot.plot(x_values, x2)
        plot.xlabel("Tamaño")
        plot.ylabel("Crecimiento")
        plot.legend()
        plot.show()