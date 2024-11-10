import json

# functia read_data() citeste articolele din json-ul articole.json
# si face un dictionar de forma sport_dict[nume_sport] = [ cuvinte ]
# unde nume_sport este numele fiecarui sport care se poate regasi in
# json-ul citit (in cazul acesta 'football', 'basketball', 'handball',
# 'tennis' si [ cuvinte ] este lista tuturor cuvintelor din toate
# articolele corespunzatoare unui sport (cuvintele nu sunt unice - cred)
def read_data():
    fin = open('articole.json', 'r', encoding='utf-8')
    data = json.load(fin)
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

read_data()