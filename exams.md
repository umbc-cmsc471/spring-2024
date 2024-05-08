---
layout: page
title: Exams
nav_exclude: false
# permalink: /:path/
---

This page will be updated shortly before the midterm and final exams to reflect what we actually covered this semester.

# Final exam
## <span style="color:blue">Reading Guide: </span>

1. The exam will be based on the concepts, techniques and algorithms discussed in our text books (AIMI 4th edition) and in class. **Alpha-beta pruing won't be part of final exam.**
2. Reading guide covers 4-5 topics, but also the subtopics, hence long.
2. The exam will mostly focus on topics covered after Midterm. It may reference materials covered before midterm in the questions, proportional to discussions in class. 
3. The exam will be open-note. You can have the following open during exam:
- Class lectures from the website
- One page (double-sided) cheat sheet
- NO GOOGLE / WWW SEARCH

<!-- , in the same manner as they were relevant during current topic discussions in lectures.. The final will be comprehensive with more emphasis on material since the midterm exam. Review the slides we showed in class, the homework assignments, and the sample exams. -->

### Chapter 7, 8: Logical Agents 7.1-7.7; 8.1-8.3; 9.1;

<!-- - Understand how an agent can use logic to model its environment, make decisions and achieve goals, e.g. a player in the Wumpus World -->
- Understand the syntax and semantics of propositional logic
- Understand the concept of a model for a set of propositional sentences/formulas
- Understand the concept of a valid sentence (tautology) and an inconsistent sentence
- Know how to find all models entailed by a set of formulas
- Understand the resolution inference rule and how a resolution theorem prover works
- Understand the concepts of soundness and completeness for a logic reasoner
- Know what a Horn clause is in propositional logic, how to determine if a proposition sentence is a Horn clause and why Horn clauses are significant
<!-- - Conjunctive Normal Form -->
- know how to convert a set of formulas/sentences to **conjunctive normal form** and then to use resolution / refutation to try to prove if an additional formula is true
- Know what it means for a KB to be satisfiable
- Understand the limitation of propositional logic as a representation and reasoning tool
####  **First order logic**
- Understand first order logic (FOL), it's notation(s) and quantifiers
- Understand how FOL differs from higher-order logics
- Be able to represent the meaning of an English sentence in FOL and to paraphrase a FOL sentence in English

### Chapter 11: Planning (without uncertainty) -- RN 11.1, 11.2-11.2.1, 11.3; 
<!-- PM 6.2, 6.3 -->

- Understand the blocks world domain
- Understand classical strips planning
- State-space planning algorithms and heuristics 
<!-- - Be familiar with the PDDL planning representation -->

### Chapter 17: Reinforcement Learning (Planning with uncertainty) -- RN 17.1, 17.2.1
- Markov Decision Process: Formalizing Reinforcement Learning
- Expected Discounted Reward
- Calculating Value Function with Dynamic Programming and how optimal policy is calculated (**There will be no simulation question asked for this part of RL**)

### Chapter 12: Bayesian reasoning 12.1-12.7; 
- Joint, marginal, and conditional probability, checking independence
- Bayes rule and how it can be used (how does it reduce \#paramaeters)
- Naive Bayes model and (how does it reduce \#paramaeters)

### Chapter 13: Bayesian reasoning: RN 13.1, 13.2; PM 9.5, 10.2

- Bayesian belief networks
- Variable Elimination and Posterior Distribution calculation from it
- Maximum Likelihood Estimation (Advanced topic slides are not part of the syallabus) 
<!-- (not important) -->


### Chapter 19: Learning from Examples 19.1-4, 19.6, 19.7.5-9; 

- supervised vs. unsupervised ML
<!-- - Tasks: regression and classification -->
- Classification and Regression: linear and logistic/maxent models
- Decision trees
    - entropy and information gain
    - ID3 algorithm using information gain
    - handle overfitting, pruning
    - advantages and disadvantages of decision trees
- Support Vector Machine (SVM)  <!-- - linear separability of data -->
    - use of kernels
    - margin and support vectors
    <!-- - soft margin for allowing non-separable data -->
    - SVM performance and use
- ML methodology
    - Separate training and development, test and validation data, k-fold cross validation
    - Metrics: precision, recall, accuracy, F1; Learning curve
    - Confusion matrix, and Multi-class P/R/F
    - Macro vs Micro avg.
- Unsupervised ML
    - k-means clustering


<!-- - ML ensembling
    - bagging, various ways
    - random forest of decision trees
    - Advantages -->
<!-- - Clustering data -->
<!-- hierarchical clustering
dendogram
bottom-up agglomerative vs. top-down divisive - Tools - numpy array basics and difference between numpy and scipy --> 


### Chapter 21: Neural Networks 21.1-8; 

- Basic elements: nodes (inputs, hidden layers, outputs), connections, weights, activation function
- Types of neural networks and their advantages/disadvantages/**purpose**
    - Basic perceptron (single layer, step activation function)
    - MLP: Multi-layer perceptron
    - Feed Forward networks
    - *Use of CNN: convoluted neural network* 
    - *Use of RNN: recurrent neural network*
    - ~~Use of Transformers~~
- ~~Fine tuning and Transfer Learning~~
- Training process
    - Loss function, Backpropagation (No derivation), Activation functions (step, ReLu, sigmoid, tanh), batches and epochs, dropout
- Advantages and disadvantages of neural networks for supervised machine learning compared to other methods (e.g., decision trees, SVM)

<!-- - Awareness of tools (High-level idea)
    - Tensorflow/Pytorch vs Keras -->
<!-- ([Notes](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)) -->
<!-- - Convolution and pooling layers
        - Why we need activation?
        - What is kernel and how is it different from weights?
        - How are the final features created? -->


Here are old exams that you can use as examples of what to expect.  Some exams are courtesy of Dr. Tim Finin from his previous offerings. The content has varied over the years, so you should ignore anything that we did not cover this semester.
- [2016](https://redirect.cs.umbc.edu/courses/undergraduate/471/spring23/02/exams/f2016a.pdf) (with [answers](https://redirect.cs.umbc.edu/courses/undergraduate/471/spring23/02/exams/f2016a.pdf)) [Less emphasis on Q5]
- [2018](https://redirect.cs.umbc.edu/courses/undergraduate/471/spring23/02/exams/f2018.pdf) (with [answers](https://redirect.cs.umbc.edu/courses/undergraduate/471/spring23/02/exams/f2018a.pdf))
- **Your homeworks**


# Midterm exam

## <span style="color:blue">Reading Guide: </span>

(1) The exam will be based on the concepts, techniques and algorithms discussed in our text books (AIMI 4th edition) and in class.

(2) It's important to have read chapters 1-6 in our text. Specifically:

- 1.1-1.5
- 2.1-2.4
- 3.1-3.6
- 4.1 (not 4.2-4.5)
- 5.1-5.5, 5.7 (not 5.6)
- 6.1-6.4, 6.5.2

This will fill in some of the gaps in our class coverage and discussions and also provide more background knowledge.

(3) listed below are things you should be prepared to do.

### Chapter 1: Artificial Intelligence
Be familiar with the foundations, state of the art and both its risks and benefits.

### Chapter 2: Intelligent agents
Understand the basic frameworks and characteristics of environments and agents introduced in chapter 2.

### Chapters 3 and 4: Search

- Take a problem description and come up with a way to represent it as a search problem by developing a way to represent the states, actions, and recognize a goal state.
- Be able to analyze a state space to estimate the number of states, the 'branching factor' of its actions, etc.
- Know the properties that characterize a state space (finite, infinite, etc.) and algorithms that are used to find solutions (completeness, soundness, optimality, etc).
- Understand both uninformed and informed search algorithms including breadth first, depth first, best first, algorithm A, algorithm A*, iterative deepening, depth limited, bi-directional search, beam search, uniform cost search, etc.
- Understand local search algorithms including hill-climbing and its variants, simulate annealing and genetic algorithms, advantages and disadvantages.
<!-- Know how to simulate these algorithms. -->
- Be able to develop heuristic functions and tell which ones are admissible.
- Understand how to tell if one heuristic is more informed than another.

### Chapter 6: Constraint Satisfaction Problems

- Understand the basics of CSP, including variables, domains, constraints.
- Be able to take a problem description and set it up as a CSP problem. For example, identify a reasonable set of variables, indicate the domain of each, describe the constraints that must hold on variables or sets of variables.
- Understand the forward checking and ARC-3 algorithms and be able to simulate them.
- Understand the MRV and LCV algorithm and be able to simulate it.
- Understanding of Tree-Structured CSPs.

### Chapter 5: Adversarial search and Game theory

- Understand the basic characteristics of games
- Understand and be able to simulate Minimax with and without alpha-beta given a game tree and the values of the static evaluation function on the leaves.
<!-- - Be able to take a game and develop a representation for it and its moves and to describe a reasonable static evaluation function. -->
- Understand how to handle games with uncertainty.
- Understanding of MCTS algorithm

## <span style="color:blue">Sample Questions </span>
I am providing some sample exams. Some exams are courtesy of Dr. Tim Finin from his previous offerings. Please ignore the questions about game theory. Some old midterms from CMSC 471 course:

- [2023-fall-midterm-exam](assets/exams/mt23f.pdf) with [solution](assets/exams/mt23fa.pdf)
<!-- - [2019-midterm-exam](https://redirect.cs.umbc.edu/courses/undergraduate/471/spring21/02/exams/mt19.pdf) with [solution](assets/midterm/mt19a.pdf) -->
- [2018-midterm-exam](https://redirect.cs.umbc.edu/courses/undergraduate/471/spring21/02/exams/mt18.pdf) with [solution](assets/exams/mt18a.pdf)


