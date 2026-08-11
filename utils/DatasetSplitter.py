import random;

class DatasetSplitter:

    @staticmethod
    def split(samples):

        # fijamos una semilla del generador de números aleatorios.
        random.seed(42)
        random.shuffle(samples)

        # separamos la muestra para obtener un 2/3 para el entrenamiento y 1/3 para verificar
        separator = int(len(samples) * 2/3)

        training = samples[:separator]
        validation = samples[separator:]

        return training, validation

