import random;

class Perceptron:

    def __init__(self, input_size):

        # fijamos una semilla para obtener siempre los mismos pesos iniciales
        random.seed(10)

        self.learning_rate = 0.1
        self.bias = 1
        self.threshold = 0
        self.weights = [
            random.uniform(-1,1) 
            for _ in range(input_size)
            ]
        self.training_errors = []
        self.training_history = []

    def postsynaptic_potential(self, features):
        """
        Funcion para calcular el potencial postisnaptico del perceptrón dado por la regla de propagación psp = x * w + b
        donde x es una entrada, w es el peso asociado y b es el bias.
        """
        psp = self.bias

        for feature, weight in zip(features, self.weights):
            psp += feature * weight

        return psp

    def activation_func(self, psp):
        """
        Funcion que calcula la activación de la neurona basandose en el umbral de activación.
        Si el potencial postsináptico supera o iguala el umbral, la función retorna 1 (bacteria maligna). Caso contrario, retorna 0 (bacteria benigna).
        """
        if psp >= self.threshold:
            return 1
        else: 
            return 0

    def predict(self, features):
        psp = self.postsynaptic_potential(features)
        return self.activation_func(psp)    

    def calculate_error(self, expected, predicted):
        """
        Función para calcular la tasa de error basado en la técnica de aprendizaje supervisado de corrección del error.
        ERROR = salida deseada - salida obtenida
        """
        return expected - predicted

    def update_weights(self, features, error):
        """
        Función que calcula el valor nuevo de cada peso asociado a cada entrada para ajustar la precisión del peceptrón a la hora de clasificar los datos.
        Actualiza el bias en base a el error calculado en calculate_error y la tasa de aprendizaje.
        """

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (self.learning_rate * error * features[i])

        self.bias = self.bias + self.learning_rate * error

    def train(self, training_samples):
        """
        Función que entrena al perceptrón con la cantidad de datos de entrenamiento equivalentes a 2/3 del total del conjunto en una cantidad determinada de epocas.
        """

        era = 0

        while (era < 1000):

            for bacterium in training_samples:
                result = self.predict(bacterium.features)
                error = self.calculate_error(bacterium.expected , result)

                if error != 0:
                    self.update_weights(bacterium.features, error)

            accuracy = self.training_accuracy(training_samples)
            self.training_history.append(accuracy)

            if accuracy >= 95:
                print(
                    f"Entrenamiento finalizado en la época {era} "
                    f"con una precisión de {accuracy:.2f}%"
                )
                break

            era += 1

    def training_accuracy(self, training_samples):
        """
        Función que calcula la cantidad de bacterias del conjunto de entrenamiento que siguen siendo clasificadas incorrectamente.
        """

        correct_predict = 0

        for bacterium in training_samples:

            result = self.predict(bacterium.features)

            if result == bacterium.expected:
                correct_predict += 1

        accuracy = (correct_predict / len(training_samples)) * 100

        return accuracy

    def validate(self, validation_samples):
        """
        Función para validar que el perceptrón clasifique correctamente el conjunto de datos. 
        Utiliza el 1/3 restante de los datos.
        Se lo considera apto para operar si clasifica bien más del 95% de los datos.
        """

        correct_predict = 0
        failed_samples = []

        print("\nResultados de la validación:")
        print("-" * 70)
        print(f"{'ID'} {'Tamaño':>10} {'Crecimiento':>15} {'Esperado':>12} {'Obtenido':>12}")
        print("-" * 70)

        for bacterium in validation_samples:
            result = self.predict(bacterium.features)

            if bacterium.expected == result:
                correct_predict += 1
            else:
                failed_samples.append(bacterium)

            print(
                f"{int(bacterium.ids):3d}"
                f"{bacterium.features[0]:10.2f}"
                f"{bacterium.features[1]:15.2f}"
                f"{int(bacterium.expected):12d}"
                f"{result:12d}"
                )

        accuracy = (correct_predict / len(validation_samples)) * 100

        return accuracy, correct_predict, failed_samples