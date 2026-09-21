# minotaur-chess-ai
The Minotaur chess AI project aims to create a new framework for chess analysis. It is fundamentally a representations first approach to chess that represents the chess board as a [heterogenous graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) with different edge types for each interaction to create a high dimensional [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) chess position encoder using [Graph Attention Networks (GATs)](https://en.wikipedia.org/wiki/Graph_neural_network#Graph_attention_network), [learnable graph unpooling](https://arxiv.org/abs/2206.01874), [Deep Graph InfoMax (DGI)](https://arxiv.org/html/1809.10341v2), contrastive learning via [InfoNCE](https://arxiv.org/abs/2407.00143), and [reinforcement learning via self-play](https://en.wikipedia.org/wiki/Self-play). The idea is that a sufficiently advanced and thorough position encoding could contain emergent organizational properties that enable easier and higher quality downstream position analysis, whether that be a chess playing AI or a classifier of some kind.

## Table of Contents

* Labyrinth (Encoder)
   * Architecture / training
* Minotaur (Player)
   * Architecture
   * Pre-training
   * Supervised learning
   * Reinforcement learning
   * Adversarial model


## Labyrinth (Encoder):
#### Architecture / training:
Labyrinth is the position encoder that will be trained fully self supervised using Deep Graph InfoMax and contrastive learning via InfoNCE. DGI involves three agents working together in a single training loop: the local encoder, the global summarizer, and the discriminator.

The local encoder involves three graph attention network layers on the heterogenous graph which ensures that every single node has received information from every other node through multiple paths, as well as [multilayer perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron). This results in 64 different encoded vectors, and each one essentially describes the relationships between that square and every other square on the board.

The global summarizer involves multiple learnable graph unpooling layers that learn to expand the position from 64 nodes into a larger graph that makes explicit some of the deeper relationships that were previously only implicit in the position. The larger concept graph is then fed into a graph attention network then flattened and fed into a multilayer perceptron before outputting one large summary vector.

The discriminator must be able to look at the highly descriptive summary vector and determine if a specific node encoding belongs to that summarized position or if it came from a perturbed position. In other words, if a perturbed position had a pawn removed from the g2 square, then the discriminator should be able to look at the summary of the original position, and the encoding of the b8 square from the perturbed position, and realize that they do not match. This should force the summarizer and local encoder to create highly descriptive encodings.

Somewhat unconventionally, the global summarizer will be kept as the final position encoder, since it summarizes the entire position so descriptively as to recognize the nuances of the relationships of every square it contains.

## Minotaur (Player):
#### Architecture:
After a chess position is encoded, it is passed to a [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) along with a hidden state vector. After a forward pass, the model outputs the hypothesis move, along with a modified hidden state. The hypothesis move can either be chosen, or the modified hidden state fed back into the network for another pass along with the original high-context encoded vector. This allows the model to perform a 'latent search' by continuing to think about the implications of the current position without explicitly choosing/pruning specific lines to analyze.

#### Pre-training (RL)
The model may be pre-trained on a very large collection of unlabeled chess960 positions to predict sequences of legal moves without regard to their quality. The hope is to learn extremely robust and perfectly unbiased representations for the game of chess so as to maximize the benefit of the supervised learning phase when the model learns which moves are good. By learning to predict sequences of legal unbiased moves, rather than singular moves, the model will learn to implicitly understand the consequences of moves, and drastically increase the robustness of its representations.

This will be done with reinforcement learning, where a penalty will be applied for illegal moves, and a reward for how closely the network's moves resemble a random distribution as simulated using a [Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The network will learn to make these legal but random moves using reinforcement learning. To further strengthen the robustness of its representations, I will employ [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) and [noise injection](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/). By entering the supervised learning phase with a robust and comprehensive but unbiased representation of chess, it will maximize the value of the limited data.

The model will receive a mini-batch of random chess960 positions of size M. For each position in the mini-batch, the model will output a sequence of S moves, meaning a single mini-batch involves predicting M times S moves. A Monte-Carlo simulation will generate random moves for every position in the mini-batch as well as the positions that result from the moves chosen by the model. After both of those processes are complete, the incidence of each move chosen by the AI across all positions will be compared to the the average incidence for each move across all positions in the Monte-Carlo simulation. The mini-batch will then receive a final score based on the number of chosen moves that were legal, and how closely those legal moves matched the distribution from the Monte-Carlo simulation.

The pre-training will conduct N number of mini-batches in a batch. After each batch, the action network and value networks will be updated according to Proximal Policy Optimization. A batch consists of N mini-batches, which is N times M initial positions, and up to N times M times S total chess positions.

#### Supervised learning
A collection of chess positions has been curated to be evaluated by Stockfish and LeelaChessZero (lc0). Since chess is solved for positions with 7 or fewer pieces, only positions with greater than 7 pieces will be analyzed. Endgame databases will be used to get perfect-quality data for positions with 7 or fewer pieces.

Forced checkmate sequences are another source of perfect-quality training data. A program will analyze positions to a medium-low depth to scan for positions with forced checkmates. I can then use those positions to expand a tree of forced checkmate sequences that can all be used in training. The tree will also be expanded upward to find positions that are even further from the forced checkmate. Positions that are very far from their forced checkmate, such as checkmate in 20 or greater, will be the most valuable as the AI may be able to generalize from these positions to non-forced checkmate positions.

When training, positions should maximize diversity across the embedding space to improve generalization.

#### Reinforcement learning
The model will also be trained using [Reinforcement Learning (RL) with self-play](https://en.wikipedia.org/wiki/Self-play), similar to Leela and AlphaZero. I may experiment with starting games in random positions, and choosing positions randomly distributed over the chess position embedding space. To randomly choose points in the hypersphere volume, use a [Gaussian variable](https://en.wikipedia.org/wiki/Normal_distribution) for each dimension and normalizing them to rest on the surface of a hypersphere, then randomly scaling inward proportionally to the nth root of the random factor. Once a random point is chosen, it can be decoded into a chessboard configuration, and a game can be played starting from that point. The issue remains to choose positions that are roughly equal in evaluation.

#### Adversarial model -> supervised learning
After using the previous methods (either supervised + self-play fine-tuning, or standalone self-play) I would like to train an adversarial network to learn the weaknesses of my model. This method was used to [defeat the superhuman Go AI, KataGo](https://arxiv.org/abs/2211.00241). Afterward, the American human, Kellin Pelrine, was able to learn this strategy to [defeat KataGo](https://arstechnica.com/information-technology/2023/02/man-beats-machine-at-go-in-human-victory-over-ai/). If my adversarial model is able to defeat the Minotaur model, I will then take positions from games that Minotaur lost and feed them into Stockfish 16 at high depth, and use that data to fine-tune the model with additional supervised learning.

This process will then be repeated.

