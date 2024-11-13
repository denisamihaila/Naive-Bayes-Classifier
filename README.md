![GitHub Repo stars](https://img.shields.io/github/stars/denisamihaila/Project-1-Probabilities-and-Statistics)

# Bayes Sport Classifier
This project serves to help users know what type of sport a piece of text is talking about with a **>85%** success rate, by implementing a Naive Bayes classification model in Python. The Bayes theorem is applied to compute the probability of a text belonging to each class (type of sport) based on word frequency within each sport category.

## Built with
<img src="https://img.shields.io/badge/json-5E5C5C?style=for-the-badge&logo=json&logoColor=white"/> <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/>

## Mathematical model used
### Bayes' Theorem
```math
\mathbb{P}(\text{Sport}|\text{Text})=\frac{\mathbb{P}(\text{Text}|\text{Sport})\cdot\mathbb{P}(\text{Sport})}{\mathbb{P}(\text{Text})}
```

### Laplace Smoothing
The probability calculation uses Laplace Smoothing to handle the zero probability problem. For each word, the probability of it belonging in a class is computed as follows:
```math
\mathbb{P}(\omega|S)=\frac{\text{count}(\omega|S)+1}{\text{total words in }S+|V|}
```
where:
- $\text{count}(\omega|S)$ is the frequency of a given word $\omega$ in sport $S$
- $|V|$ is the vocabulary size

### Logarithmic Transformation
To avoid underflow, probabilities are transformed into log probabilities:
```math
\log\mathbb{P}(\text{Sport}|\text{Text})=\log\mathbb{P}(\text{Sport})+\sum_{\omega\in\text{Text}}\log\mathbb{P}(\omega|\text{Sport})
```
At the end, the sport with the highest $\log\mathbb{P}(\text{Sport}|\text{Text})$ is chosen.

## Code Structure
### Modules imported
- `json`: to load and parse data from JSON files
- `numpy`: used for numerical operations, specifically `np.log()` and `np.exp()` for probability calculations

### Functions and Classes
- `read_data(file_name)`: reads JSON data, returns a dictionary of sports with associated texts
- `Bayes` class:
  - `process_data(data)`: organizes words by sport and calculates prior probability of each sport
  - `probabilities_of_words(tuple_data)`: calculates the smoothed probability of each word within each sport
  - `train(file_name)`: loads training data, process it, and calculates word probabilities
  - `prompt(text)`: classifies a given text into a sport category by calculating probabilities
  - `accuracy()`: computes the model's accuracy on a test dataset

<!---
To add: usage instructions
-->

## Limitations
The project is a WIP and at the moment only supports detection for football, basketball, handball and tennis. More sports may be added in the future upon request. The >85% model accuracy makes it an useful tool for categorizing aforementioned sports, and the accuracy may be improved in potential future updates.

## Contributing
Feel free to add to this project by forking or by opening a pull request.

## References
- Class courses and laboratories held by professor Mihai Bucătaru at the University of Bucharest
- *[An Essay Towards Solving a Problem in the Doctrine of Chances](https://bayes.wustl.edu/Manual/an.essay.pdf)* by Thomas Bayes
- *[Laplace smoothing in Naive Bayes algorithm](https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece)* by Vaibhav Jayaswal
- *[Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA)* by Josh Starmer
