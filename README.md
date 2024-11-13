# Bayes Sport Classifier
This project serves to help users know what type of sport a piece of text is talking about with a **>85%** success rate, by implementing a Naive Bayes classification model in Python. The Bayes theorem is applied to compute the probability of a text belonging to each class (type of sport) based on word frequency within each sport category.

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

