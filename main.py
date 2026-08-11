from utils.CsvReader import CsvReader;
from utils.DatasetSplitter import DatasetSplitter;
from model.Perceptron import Perceptron;
from visualization.LinearFunction import LinearFunction
from visualization.TrainingHistory import TrainingHistory

reader = CsvReader()
samples = reader.read_csv()

training, validation = DatasetSplitter.split(samples)

perceptron = Perceptron(input_size=2)

perceptron.train(training)

accuracy, correct_predict, failed_samples = perceptron.validate(validation)

print("\n")
print(f"Precisión: {accuracy}%")
print(f"Aciertos: {correct_predict}")
print(f"Fallos: {len(failed_samples)}\n")

print("Pesos finales:")
print("Pesos: ", perceptron.weights)
print("Bias: ", perceptron.bias)

if failed_samples:
    print("\nBacterias mal clasificadas:")
    for bacterium in failed_samples:
        print(
            f"ID: {bacterium.id} | "
            f"Tamaño: {bacterium.features[0]:.2f} | "
            f"Crecimiento: {bacterium.features[1]:.2f} | "
            f"Esperado: {int(bacterium.expected)}"
        )

LinearFunction.linear_func(samples, perceptron)
TrainingHistory.plot(perceptron.training_history)