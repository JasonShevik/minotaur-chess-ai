# minotaur-chess-ai
This is a chess AI that I am developing to utilize a neural network classifier to evaluate the board and determine the best next move. This AI does not use a search function to look multiple moves ahead, rather using only a single forward pass.

It will have 4 output nodes (starting row and column, ending row and column) and 64 inputs (the squares).

## Table of Contents

* In Progress
* Architecture
* Training plan
   * Pre-training
   * Supervised learning
   * Reinforcement learning
   * Adversarial model
* Reinforcement Learning Details
   * Evolutionary AI
   * Hyperspheres and randomness

## In Progress:
* Write the pre-training function and pre-train the model

## Architecture:
The model is currently programmed to simply use a densly connected neural network. I've also considered using a convolutional neural network, but haven't implemented that. These are very simple and standard ways to create a chess AI.

Recently, however, I've considered switching to a [Graph neural network (GNN)](https://en.wikipedia.org/wiki/Graph_neural_network) such as a [Graph Convolutional Network (GCN)](https://en.wikipedia.org/wiki/Graph_neural_network#Graph_convolutional_network) or a [Graph Attention Network (GAN)](https://en.wikipedia.org/wiki/Graph_neural_network#Graph_attention_network).

Rather than the input to the network being a vector of 64 values for the squares, it could be a graph with 64 nodes and 6 different edge types for each piece type which show where pieces can move. The nodes would still hold integer values from -6 to 6 showing which piece, if any, is on each square, but the AI would directly see how squares connect to each other.

There should possibly be a 7th edge type that represents en passant and castling.

## Training plan
#### Pre-training (RL)
The model will be pre-trained on a very large collection of unlabeled chess960 positions to predict sequences of legal moves without regard to their quality. The hope is to learn extremely robust and perfectly unbiased representations for the game of chess so as to maximize the benefit of the supervised learning phase when the model learns which moves are good. By learning to predict sequences of legal unbiased moves, rather than singular moves, the model will learn to implicitly understand the consequences of moves, and drastically increase the robustness of its representations.

This will be done with reinforcement learning, where a penalty will be applied for illegal moves, and a reward for how closely the network's moves resemble a random distribution as simulated using a [Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The network will learn to make these legal but random moves using [Proximal Policy Optimization (PPO)](https://en.wikipedia.org/wiki/Proximal_policy_optimization) reinforcement learning. To further strengthen the robustness of its representations, I will employ [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) and [noise injection](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/). By entering the supervised learning phase with a robust and comprehensive but unbiased representation of chess, it will maximize the value of the limited data.

The model will receive a mini-batch of random chess960 positions of size M. For each position in the mini-batch, the model will output a sequence of S moves, meaning a single mini-batch involves predicting M times S moves. A Monte-Carlo simulation will generate random moves for every position in the mini-batch as well as the positions that result from the moves chosen by the model. After both of those processes are complete, the incidence of each move chosen by the AI across all positions will be compared to the the average incidence for each move across all positions in the Monte-Carlo simulation. The mini-batch will then receive a final score based on the number of chosen moves that were legal, and how closely those legal moves matched the distribution from the Monte-Carlo simulation.

The pre-training will conduct N number of mini-batches in a batch. After each batch, the action network and value networks will be updated according to Proximal Policy Optimization. A batch consists of N mini-batches, which is N times M initial positions, and up to N times M times S total chess positions.

#### Supervised learning
A collection of chess positions will be curated (through random generation and/or randomly chosen positions in lichess chess960 games). These positions will then be evaluated by Stockfish 16.1 and LeelaChessZero (lc0) at high depth to ensure very high quality data. The engine will check the score at various depth checkpoints to determine whether it is worth analyzing deeper to save computing resources. More equal positions will be analyzed more deeply.

Since chess is solved for positions with 7 or fewer pieces, only positions with greater than 7 pieces will be analyzed. Endgame databases will be used to get perfect-quality data for positions with 7 or fewer pieces.

Forced checkmate sequences are another source of perfect-quality training data. A program will analyze positions to a medium-low depth to scan for positions with forced checkmates. I can then use those positions to expand a tree of forced checkmate sequences that can all be used in training. The tree will also be expanded upward to find positions that are even further from the forced checkmate. Positions that are very far from their forced checkmate, such as checkmate in 20 or greater, will be the most valuable as the AI may be able to generalize from these positions to non-forced checkmate positions.

#### Reinforcement learning
The model will also be trained using reinforcement learning at some point. This could be implemented a few different ways.

#### Adversarial model -> supervised learning
After using the previous methods (either supervised + self-play fine-tuning, or standalone self-play) I would like to train an adversarial network to learn the weaknesses of my model. This method was used to [defeat the superhuman Go AI, KataGo](https://arxiv.org/abs/2211.00241). Afterward, the American human, Kellin Pelrine, was able to learn this strategy to [defeat KataGo](https://arstechnica.com/information-technology/2023/02/man-beats-machine-at-go-in-human-victory-over-ai/). If my adversarial model is able to defeat the Minotaur model, I will then take positions from games that Minotaur lost and feed them into Stockfish 16 at high depth, and use that data to fine-tune the model with additional supervised learning.

This process will then be repeated.


### Reinforcement learning details
#### Evolutionary AI
As an experiment, after a base version of the AI has been trained, I will use an [evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) to train the model further. The model's weights represent a specific point in a parameter space. We define a hypersphere around the point, and instantiate a number of other models in that region of the parameter space. Those models will then compete in a large tournament, or other evaluation step, and the model with the best score will be kept. We then repeat the process of defining a hypersphere and choosing additional points within the resulting. The quality of the evaluation step is critical to avoid regressing at some skills.

#### Hyperspheres and Randomness
Choosing random points within a hypersphere more complicated than it sounds. The method of choosing can affect the distribution of the sample, similar to the [Bertrand Paradox](https://en.wikipedia.org/wiki/Bertrand_paradox_(probability)) where three different methods of choosing random lines in a circle result in three different distributions of points, such as biasing toward or against the center.

Luckily, [there are known ways to get a uniform distribution](https://mathworld.wolfram.com/HyperspherePointPicking.html). We can use Gaussian random variables to determine a random point on the surface of the hypersphere and then scale it inward toward the center. However, it needs to be noted that the [volume of a hypersphere](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) is concentrated toward the edges, so the scaling needs to be weighted to account for this.

Other methods, like randomly generating polar coordinates, or randomly choosing points on the surface of the hypersphere by doing random transformations to a unit vector before scaling inward, or much different methods, like choosing two random points on the surface, and choosing a random point on the connecting line, do not yield uniform distributions.

I would like to explore the effect that these different uniform vs non-uniform distributions have on the training process and the convergence of the model.
