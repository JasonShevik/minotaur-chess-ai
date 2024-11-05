# minotaur-chess-ai
This is a new chess AI that I am developing. It will have a innovative and unconventional architecture which combines [Graph Attention Networks (GATs)](https://en.wikipedia.org/wiki/Graph_neural_network#Graph_attention_network), [Recurrent Neural Networks (RNNs)](https://en.wikipedia.org/wiki/Recurrent_neural_network), and [Reinforcement Learning (RL) with self-play](https://en.wikipedia.org/wiki/Self-play) to learn the rules of chess from scratch and how to play the best move in a given position without ever employing an explicit search function. Instead, repeated passes of the linear network will update a hidden state which contains the model's current reasoning, and the distribution of its hypothesis moves in the embedding space from one pass to another will inform the decision of when to stop.

## Table of Contents

* In Progress
* Architecture
   * Encoder/Decoder
   * Recurrent Neural Network
* Training plan
   * Pre-training
   * Supervised learning
   * Reinforcement learning
   * Adversarial model

## In Progress:
* Move all labeled data to a SQLite database, and modify the labeler files. Threads will write to a single queue that a write thread handles.
* Finish chess graph code, including a function for adding the node feature vectors, and the visualizations.
* Update the data prep file so each square is a vector rather than a scalar.
* Update the architecture, start working on encoder.

## Architecture:
#### Encoder/Decoder:
The model is utilizes a homogeneous [message passing graph](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html) implemented via a [Graph Attention Network (GAT)](https://en.wikipedia.org/wiki/Graph_neural_network#Graph_attention_network). The input to the network is a graph with 64 nodes and multiple different edge types for piece movement types (pawn move, pawn attack, knight, bishop, rook, king, castle). The node features are 13 dimensional vectors holding a one-hot encoding of the different piece types (6 for each side), plus en passant. There will be 2 or 3 graph attention layers, each with multiple attention heads, followed by a series of linear layers.

The model will then be trained with [Deep Graph Infomax (DGI)](https://arxiv.org/abs/1809.10341), a form of [Self-Supervised Learning (SSL)](https://en.wikipedia.org/wiki/Self-supervised_learning) where the model infers a hyperdimensional embedding space for chess positions by maximizing the mutual information between the embeddings of nodes or neighborhoods compared to a summary of the entire graph. The summary will be created using [DiffPool](https://arxiv.org/abs/1806.08804), a method which learns hierarchical representations of graphs by clustering the nodes and/or edges. The coarser, clustered graph will then be fed to linear layers to output a large vector which represents a point in the latent space. 

The full process involves taking a chess position (P_0) and creating a slightly purturbed copy of it (P_1), then feeding each of those to the DGI encoder model to get node level encodings that contain information about each square's neighborhood. The original position P_0 is also fed to the DiffPool summarizer model to create the latent space representation of the position. Finally, a discriminator classifier model takes in the full board summary along with a node level encoding and returns whether the encoded node came from that position or a purturbed position that it never sees. This involves a multi-agent learning setup where the DGI encoder, DiffPool summarizer, and discriminator all learn from each other simultaneously. Once this has been done for hundreds of millions of fischer random positions, the trained DiffPool summarizer is the chess position encoder which defines the high dimensional latent space.

#### Recurrent Neural Network (RNN):
After a chess position is encoded, it is passed to a [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) along with another vector populated by zeros which is the hidden state. After a forward pass, the model outputs another high-context embedded vector, which is the hypothesis move, along with a modified hidden state. The hypothesis move can either be chosen, or the modified hidden state fed back into the network for another pass along with the original high-context encoded vector. This allows the model to perform an 'implicit search' by continuing to think about the implications of the current position without explicitly choosing/pruning specific lines to analyze.

## Training plan
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

