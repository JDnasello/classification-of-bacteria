import pandas as pd;
from model.Bacterium import Bacterium

class CsvReader:

    def read_csv(samples):

        bacteriums_file = pd.read_csv("data/bacterias.csv")

        samples = []

        for _, row in bacteriums_file.iterrows():
            ids = row["id"]
            features = [row["tamano_um"], row["crecimiento_hora"]]
            expected = row["clase"]

            bacterium = Bacterium(ids, features, expected)

            samples.append(bacterium)

        return samples

