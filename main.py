import json
import numpy as np

def read_data(file_name):
    fin = open(file_name, 'r', encoding='utf-8')
    data = json.load(fin)
    fin.close()

    texts = dict()

    for sport in data:
        texts[sport] = list()

        for field in data[sport]:
            text = field["text"].lower()
            texts[sport].append(text)

    return texts

class Bayes:
    delimiters = [",", ";", " ", "'", "’", "-", "."]
    probability_of_a_sport = 0
    sports = []
    probability_of = []

    def process_data(self, data):
        self.sports = [sport for sport in data]
        self.probability_of_a_sport = 1 / len(self.sports)

        organized_words_by_sport = dict()
        all_words = set()

        for sport in self.sports:
            organized_words_by_sport[sport] = list()

            for field in data[sport]:
                for delimiter in self.delimiters:
                    field = field.replace(delimiter, " ")

                field = field.split()
                words = [word.lower() for word in field if len(word) > 2]
                organized_words_by_sport[sport].extend(words)
                all_words.update(words)

        return organized_words_by_sport, all_words

    # Functia asta imi intoarce o structura de date
    # din care pot lua probabilitatea unui cuvant
    # in cadrul unui sport in O(1)
    # exemplu: probability_of_word["Manchester"]["football"] = 0.013
    def probabilities_of_words(self, tuple_data):
        organized_words_by_sport, all_words = tuple_data # unpacking ...
        probability_of = {word: dict() for word in all_words}

        for word in all_words:
            for sport in self.sports:
                # Numar de cate ori apare un cuvant din TOATA lista de cuvinte disponibile
                # in cadrul fiecarui sport (tennis, football, handball, basketball)
                word_frequency = organized_words_by_sport[sport].count(word)

                # Probabilitatea unui cuvant este data de (de_cate_ori_a_aparut_acel_cuvant + 1) / (toate cuvintele din acel sport)
                # + 1 ca sa nu avem probabilitate 0
            probability_of[word][sport] = (word_frequency + 1) / (len(organized_words_by_sport[sport]))
        return probability_of

    def train(self, file_name):
        data_from_json = read_data(file_name)
        processed_data = self.process_data(data_from_json) # (organized_words_by_sport, all_words)
        self.probability_of = self.probabilities_of_words(processed_data)

    def prompt(self, text):
        for delimiter in self.delimiters:
            text = text.replace(delimiter, " ")
        words = [word.lower() for word in text.split() if len(word) > 2]

        # Formula lui Bayes explicata: https://www.youtube.com/watch?v=O2L2Uv9pdDA
        # aici se va calcula probabilitatea fiecarui sport
        # initial fiecare sport are probabilitatea 1/4

        sport_probability = {sport: np.log(self.probability_of_a_sport) for sport in self.sports} # Folosim log ca sa evitam underflow-ul
        # la final alegem sportul cu probabilitatea cea mai mare

        for word in words:
            if word in self.probability_of:
                for sport in self.sports:
                    # inmultim probabilitatea unui cuvant care se regaseste in sportul respectiv
                    # astfel se va face P(sport) * P(cuvant1 | sport) * P(cuvant2 | sport) ...
                    sport_probability[sport] += np.log(self.probability_of[word][sport]) # Folosim log ca sa evitam underflow-ul

        best_sport = max(sport_probability, key=sport_probability.get)
        best_prob = sport_probability[best_sport]

        return np.exp(best_prob), best_sport

    def accuracy(self):
        data = read_data('set_de_test.json')

        total = 0
        correct = 0

        for sport in data:
            for text in data[sport]:
                result = bayes.prompt(text)
                if result[1] == sport:
                    correct += 1
                total += 1

        return correct / total

bayes = Bayes()
bayes.train('set_de_antrenament.json')
print(f"Acuratetea modelului este de {bayes.accuracy()*100:.2f}%")