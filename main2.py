import json

delimiters = [",", ";", " ", "'", "’", "-", "."]
probability_of_a_sport = 0
sports = []

# Functia citeste datele din articole.json
# Prelucreaza datele si la final returneaza un tuplu
# organized_words_by_sport este un dictionar de forma organized_words_by_sport[nume_sport] = [cuvintele din sportul respectiv]
# all_words este o lista cu TOATE cuvintele din toate articolele
def read_data():
    fin = open('articole.json', 'r', encoding='utf-8')
    data = json.load(fin)
    fin.close()

    global sports
    sports = [sport for sport in data]

    organized_words_by_sport = dict()
    all_words = set()

    for sport in sports:
        organized_words_by_sport[sport] = list()
        for field in data[sport]:
            text = field["text"]
            for delimiter in delimiters:
                text = text.replace(delimiter, " ")
            text = text.split()

            for word in text:
                organized_words_by_sport[sport].append(word.lower())
                all_words.add(word.lower())

    global probability_of_a_sport
    probability_of_a_sport = 1 / len(sports)

    return (organized_words_by_sport, all_words)

# Functia asta imi intoarce o structura de date
# din care pot lua probabilitatea unui cuvant
# in cadrul unui sport in O(1)
# exemplu: probability_of_word["Manchester"]["football"] = 0.013
def probabilities_of_words(tpl):
    organized_words_by_sport = tpl[0]
    all_words = tpl[1]

    probability_of_word = { word: dict() for word in all_words }

    for word in all_words:
        for sport in organized_words_by_sport:
            # Numar de cate ori apare un cuvant din TOATA lista de cuvinte disponibile
            # in cadrul fiecarui sport (tennis, football, handball, basketball)
            word_frequency = organized_words_by_sport[sport].count(word)

            # Probabilitatea unui cuvant este data de (de_cate_ori_a_aparut_acel_cuvant + 1) / (toate cuvintele din acel sport)
            # + 1 ca sa nu avem probabilitate 0
            probability_of_word[word][sport] = (word_frequency + 1) / (len(organized_words_by_sport[sport]))

    return probability_of_word

def classify_prompt(prompt, probability_of):
    global sports, probability_of_a_sport

    for delimiter in delimiters:
        prompt = prompt.replace(delimiter, " ")
    words = [word.lower() for word in prompt.split()]

    result = (0, "")

    for word in words:
        for sport in sports:
            if word in probability_of:
                probability = probability_of_a_sport * probability_of[word][sport]
                if probability > result[0]:
                    result = (probability, sport)

    return result

def calculate_accuracy():
    tpl = read_data()
    organized_words_by_sport = tpl[0]

    probability_of = probabilities_of_words(tpl)

    count = 0
    for sport in organized_words_by_sport:
        count += len(organized_words_by_sport[sport])

    correct = 0
    for sport in organized_words_by_sport:
        for text in organized_words_by_sport[sport]:
            correct += classify_prompt(text, probability_of)[1] == sport

    print(f"Acuratetea modelului este de {correct / count * 100:.2f}%.")

calculate_accuracy()