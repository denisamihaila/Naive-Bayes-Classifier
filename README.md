![GitHub contributors](https://img.shields.io/github/contributors/denisamihaila/Naive-Bayes-Classifier?style=flat) ![GitHub Repo stars](https://img.shields.io/github/stars/denisamihaila/Naive-Bayes-Classifier?style=flat) ![GitHub forks](https://img.shields.io/github/forks/denisamihaila/Naive-Bayes-Classifier?style=flat) ![GitHub top language](https://img.shields.io/github/languages/top/denisamihaila/Naive-Bayes-Classifier?style=flat) ![GitHub License](https://img.shields.io/github/license/denisamihaila/Naive-Bayes-Classifier?style=flat)

# Naive Bayes Classifier
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
## Requirements
### Git
#### Windows
Use the install instructions provided on the [git-scm website](https://git-scm.com/downloads/win).
#### MacOS
Use [Homebrew](https://brew.sh/) or your preferred package manager. Homebrew example:
```bash
brew install git
```
#### Linux
Use your distro's package manager to install git.
##### Debian
```bash
sudo apt update &&
sudo apt install git
```
##### Arch
```bash
sudo pacman -Syu &&
sudo pacman -S git
```
#### Fedora
```bash
sudo dnf install git
```
#### openSUSE
```bash
sudo zypper install git
```
#### Other distributions
For other distributions, refer to the official package manager documentation to install git.

### Python
#### Windows
Download from [the official website](https://www.python.org/downloads/windows/) and follow the install instructions
#### MacOS
Use [Homebrew](https://brew.sh/) or your preferred package manager. Homebrew example:
```bash
brew install python
```
#### Linux
Use your distro's package manager to install Python.
##### Debian
```bash
sudo apt update &&
sudo apt install python3 python3-venv python3-pip
```
##### Arch
```bash
sudo pacman -Syu &&
sudo pacman -S python python-pip python-virtualenv
```
#### Fedora
```bash
sudo dnf install python3 python3-pip python3-virtualenv
```
#### openSUSE
```bash
sudo zypper install python3 python3-pip python3-virtualenv
```
#### Other distributions
For other distributions, refer to the official package manager documentation to install Python 3 and its required packages.

## Usage
### Cloning the repository
```bash
git clone https://github.com/denisamihaila/Naive-Bayes-Classifier &&
cd Naive-Bayes-Classifier
```
### Running the code
Use the command line, or your preferred IDE or code editor to run the python code. The code will prompt you to input your desired action:
```console
==== Naive Bayes Sport Classifier ====
---- Choose an option (1/2) from the list below ----
 =>  1. Check model accuracy
 =>  2. Enter a prompt
     Enter your option here:
```
#### Checking model accuracy
To check the model accuracy, simply input the value 1:
```console
==== Naive Bayes Sport Classifier ====
---- Choose an option (1/2) from the list below ----
 =>  1. Check model accuracy
 =>  2. Enter a prompt
     Enter your option here: 1
 ->  The model's accuracy is 86.11%.
```
#### Testing own text
To test your own piece of article or text, input the value 2, then the text you want to test:
```console
==== Naive Bayes Sport Classifier ====
---- Choose an option (1/2) from the list below ----
 =>  1. Check model accuracy
 =>  2. Enter a prompt
     Enter your option here: 2
     Your prompt: Many consider Simona Halep to be one of the greatest Romanian athletes of the 21st century.
 ->  Your prompt could be classified as a tennis text.
```

## Limitations
The project is a WIP and at the moment only supports detection for football, basketball, handball and tennis. More sports may be added in the future upon request. The >85% model accuracy makes it an useful tool for categorizing aforementioned sports, and the accuracy may be improved in potential future updates.

## References
- Class courses and laboratories held by professor Mihai Bucătaru at the University of Bucharest
- *[An Essay Towards Solving a Problem in the Doctrine of Chances](https://bayes.wustl.edu/Manual/an.essay.pdf)* by Thomas Bayes
- *[Laplace smoothing in Naive Bayes algorithm](https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece)* by Vaibhav Jayaswal
- *[Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA)* by Josh Starmer
