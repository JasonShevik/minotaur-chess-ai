# minotaur-chess-ai
This is a chess AI that I am developing to utilize a neural network to evaluate the board and determine the best next move. This AI does not use a search function to look multiple moves ahead.

Since the model will not calculate ahead, it will be a classifier model rather than a regression model. It will have 4 output nodes (starting row and column, ending row and column) and 66 inputs (64 squares, right to castle, en passant).

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
* Write a function to convert the [FEN format](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) positions into the 74 inputs for the neural net.
* Incorporate [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) into the learning methods.
* Incorporate [adding nosie](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/) into the learning methods.
* Write the neural net.


## Training plan
#### Pre-training (RL)
The model will be pre-trained on a very large collection of unlabeled chess960 positions to predict legal moves without regard to their quality. This will be done with reinforcement learning, where a penalty will be applied for illegal moves, and a reward for legal moves. The model will also be encouraged to have randomly distributed moves, probably by seeing how closely its choices match a monte carlo simulation.

I may also experiment with multi-output pre-training. Rather than having 4 output nodes to predict a single move like it will when playing games normally, during pretraining, I could train it to have 8, 12, 16+ nodes where I decide that some of the nodes correspond to additional legal moves in the position, or possibly legal moves in subsequent positions. By training the model to predict unbiased random legal sequences of moves in each pass on hundreds of millions of positions, the model will learn robust representations and features to represent the problem spaces. This will maximize the value of the supervised learning phase. I would also like to employ dropout and noise injection during this phase to improve robustness. It could also have output nodes that predict sequences of moves that could have preceded this position. To test whether that is actually helpful to learning, I could train one that predicts -3 to +3 moves from now versus one that predicts +6 moves from now, then do the same supervised learning step on both.

#### Supervised learning
A collection of chess positions will be curated (through random generation and/or randomly chosen positions in lichess chess960 games). These positions will then be evaluated by Stockfish 16 at high depth to ensure very high quality data. The Stockfish evaluation depth will be determined by the evaluation of the position at depth 20. The more even the position, the higher the depth, up to a depth of 50 or 60 for the most equal positions. This is to optimize the data labeling process, since close positions are more critical, and I don't want to waste additional compute power on clearly won/lost positions.

Since chess is solved for positions with 7 or fewer pieces, I will only use Stockfish to evaluate positions with greater than 7 pieces. Endgame databases will be used to get high quality labeled data for positions with 7 or fewer pieces.

I would also like to collect a large collection of data on forced checkmate sequences, since this also represents perfect-quality training data. By going through positions and collecting a database of forced checkmate positions, I can then search through all of the different variations, including trying to work backwards toward even longer forced checkmate sequences, and generate hundreds of training examples for each position I initially collect. To what extent would the AI be able to generalize from this data? Expecially if I can get extremely long sequences of 30 moves or more.

#### Reinforcement learning
The model will also be trained using reinforcement learning at some point. This could be implemented a few different ways

#### Adversarial model -> supervised learning
After using the previous methods (either supervised + self-play fine-tuning, or standalone self-play) I would like to train an adversarial network to learn the weaknesses of my model. This method was used to [defeat the superhuman Go AI, KataGo](https://arxiv.org/abs/2211.00241). Afterward, the American human, Kellin Pelrine, was able to learn this strategy to [defeat KataGo](https://arstechnica.com/information-technology/2023/02/man-beats-machine-at-go-in-human-victory-over-ai/). If my adversarial model is able to defeat the Minotaur model, I will then take positions from games that Minotaur lost and feed them into Stockfish 16 at high depth, and use that data to fine-tune the model with additional supervised learning.

This process will then be repeated.


### Reinforcement learning details
#### Evolutionary AI
I'm not sure how effective this will be, but as an experiment, after a base version of the AI has been trained, I will use an [evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) to train the model further. The model's weights represent a specific point in a parameter space. We define a hypersphere around the point, and instantiate a number of other models in that region of the parameter space. Those models will then compete in a large tournament, and the model with the best score will be kept. We then repeat the process of defining a hypersphere and choosing additional points within the resulting. 

#### Hyperspheres and Randomness
Choosing random points within a hypersphere is a bit more complicated than it sounds. The method of choosing can affect the distribution of the sample, similar to the [Bertrand Paradox](https://en.wikipedia.org/wiki/Bertrand_paradox_(probability)) where three different methods of choosing random lines in a circle result in three different distributions of points, such as biasing toward or against the center.

Luckily, [there are known ways to get a uniform distribution](https://mathworld.wolfram.com/HyperspherePointPicking.html). We can use Gaussian random variables to determine a random point on the surface of the hypersphere and then scale it inward toward the center. However, it needs to be noted that the [volume of a hypersphere](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) is concentrated toward the edges, so the scaling needs to be weighted to account for this.

Other methods, like randomly generating polar coordinates, or randomly choosing points on the surface of the hypersphere by doing random transformations to a unit vector before scaling inward, or much different methods, like choosing two random points on the surface, and choosing a random point on the connecting line, do not yield uniform distributions.

I would like to explore the effect that these different uniform vs non-uniform distributions have on the training process and the convergence of the model.
