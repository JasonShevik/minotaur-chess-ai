# minotaur-chess-ai
This is a chess AI that I am developing to utilize a neural network classifier to evaluate the board and determine the best next move. This AI does not use a search function to look multiple moves ahead, rather using only a single forward pass.

It will have 4 output nodes (starting row and column, ending row and column) and 64 inputs (the squares).

## Table of Contents

* To Do
* Training plan
   * Pre-training
   * Supervised learning
   * Reinforcement learning
   * Adversarial model
* Reinforcement Learning Details
   * Evolutionary AI
   * Hyperspheres and randomness


## To Do:
* Write the pre-training function and pre-train the model
* Incorporate [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) into the learning methods.
* Incorporate [adding nosie](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/) into the learning methods.


## Training plan
#### Pre-training (RL)
The model will be pre-trained on a very large collection of unlabeled chess960 positions to predict legal moves without regard to their quality. This will be done with reinforcement learning, where a penalty will be applied for illegal moves, and a reward for legal moves. The model will also be encouraged to have randomly distributed moves, probably by seeing how closely its choices match a monte carlo simulation. Ideally, the model will be trained to predict long sequences of randomly distributed legal moves so that it implicitly learns about the consequences of moves. To further strengthen the robustness of its representations, I will employ dropout and noise injection. By entering the supervised learning phase with a robust and comprehensive but unbiased representation of chess, it will maximize the value of the limited data.

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
