import json
from collections import Counter

# functia read_data() citeste articolele din json-ul articole.json
# si face un dictionar de forma sport_dict[nume_sport] = [ cuvinte ]
# unde nume_sport este numele fiecarui sport care se poate regasi in
# json-ul citit (in cazul acesta 'football', 'basketball', 'handball',
# 'tennis' si [ cuvinte ] este lista tuturor cuvintelor din toate
# articolele corespunzatoare unui sport (cuvintele nu sunt unice - cred)
def read_data():
    fin = open('articole.json', 'r', encoding='utf-8')
    data = json.load(fin)
    fin.close()

    sport_dict = dict()
    delimiters = [",", ";", " ", "'", "’", "-", "."]
    for sport in data:
        all_words = []
        for articol in data[sport]:
            text = articol["text"]
            for delimiter in delimiters:
                text = " ".join(text.split(delimiter))
            words = [word.lower() for word in text.split() if len(word) > 2]
            all_words.extend(words)

        sport_dict[sport] = all_words

    return sport_dict

# functia primeste dictionarul cu cuvinte
# si calculeaza probabilitatea fiecare cuvant
# sa apara intr-o categorie de sport
def calculate_probabilities_word(sport_dict):
    word_prob = dict()
    for sport in sport_dict: # football, handball, tennis, basketball
        word_prob[sport] = dict()
        total_words = len(sport_dict[sport])
        used_words = set()

        for word in sport_dict[sport]:
            if word not in used_words:
                appearances = sport_dict[sport].count(word)
                word_prob[sport][word] = appearances / total_words
                used_words.add(word)
    return word_prob

def classify_text(text, probs):
    delimiters = [",", ";", " ", "'", "’", "-", "."]
    for delimiter in delimiters:
        text = text.replace(delimiter, " ")
    words = [word.lower() for word in text.split() if len(word) > 2]

    # P(category)
    p_category = 1 / len(probs)

    result = (0, "")

    for category in probs:

        # Compute P(word)
        P_word = 0.0001
        for word in words:
            P_word += words.count(word) / len(words)

        # Compute P(word | category)
        P_word_given_category = 0.0001
        for word in words:
            if word in probs[category]:
                P_word_given_category += probs[category][word]

        # Compute P(category | word)
        p_category_given_word = P_word_given_category * p_category / P_word

        if p_category_given_word >= result[0]:
            result = (p_category_given_word, category)

    return result

data = read_data()
probs = calculate_probabilities_word(data)
res = classify_text("I've won this tennis match using my FIFA Manchester City team", probs)
print(f"Textul apartine categoriei {res[1]} avand probabilitatea de {res[0] * 100:.2f}%.")