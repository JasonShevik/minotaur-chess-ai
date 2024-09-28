# minotaur-chess-ai
This is a chess AI that I am developing for a neural network classifier to evaluate the board and determine the best next move. This AI does not use a search function to look multiple moves ahead, rather using only a single forward pass.

It will have 4 output nodes (starting row and column, ending row and column) and a graph input.

## Table of Contents

* In Progress
* Architecture
* Training plan
   * Pre-training
   * Supervised learning
   * Reinforcement learning
   * Adversarial model
* Encoder / Embedding

## In Progress:
* Write the pre-training function and pre-train the model
* Move all labeled data to a SQLite database, and modify the labeler files. Threads will write to a single queue that a write thread handles.
* Finish chess graph code, including a function for adding the node feature vectors, and the visualizations.

## Architecture:
The model is utilizes a homogeneous [message passing graph](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html) implemented via a [Graph Attention Network (GAT)](https://en.wikipedia.org/wiki/Graph_neural_network#Graph_attention_network).

The input to the network is a graph with 64 nodes and multiple different edge types for piece movement types. The node features are 13 dimensional vectors holding a one-hot encoding of the different piece types, plus en passant.

There will be 2 or 3 graph attention layers, each with multiple attention heads, followed by a series of linear layers.

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


## Encoder / Embedding
A separate AI encoder model will eventually be made using [Deep Graph Infomax](https://arxiv.org/abs/1809.10341), a form of [Contrastive Self-Supervised Learning](https://en.wikipedia.org/wiki/Self-supervised_learning#Contrastive_self-supervised_learning). This model will take the graph representation of chess positions as input, and output a vector in a higher dimensional place where spacial relationships between vectors correspond to meaningful similarities between positions. Most other methods, like GraphCL, organize positions based on if the boards appear visually similar, but this method should organize them based on deeper meanings and patterns.

The purpose of this model is not to serve as part of the final AI. Its only purpose is to create the embedding, so that I can use the embedding to ensure maximum data diversity in training data sets. I will select positions to label and train off of based on their spacial distribution within the embedding space to maximize spread. Balancing the data according to this metric will be significantly better than traditional balancing—such as by opening, middle game, and endgame—because those traditional methods are biased by human ideas of game phases.
