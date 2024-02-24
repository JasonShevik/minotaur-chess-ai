# minotaur-chess-ai
This is a chess AI that I am developing. It utilizes a neural network to evaluate the board and determine the best next move. This AI does not use a search function to look multiple moves ahead. I'm interested to see how powerful I can make this bot without ever calculating ahead.

Since the model will not calculate ahead, it will be a classifier model rather than a regression model. It will have two output nodes with one for the starting square and one for the ending square which together describe a single move (there will be a special case for castling).

## To Do:
* Now that the "hopeless" mechanic has been removed (Previously, if a position had a high enough score, the analysis would stop before reaching the first depth threshhold. Now, the analysis only stops early if the engine stops making progress or finds a forced checkmate.), I will need to write a function to go over all analyzed positions with low depth and re-analyze them.
* Write a function to convert the [FEN format](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) positions into the 74 inputs for the neural net.
* Incorporate [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) into the learning methods.
* Incorporate [adding nosie](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/) into the learning methods.

## Training plan
#### Pre-training (RL)
The model will be pre-trained on a very large collection of unlabeled chess960 positions to predict legal moves without regard to their quality. This will be done with reinforcement learning, where a penalty will be applied for illegal moves, and a reward for legal moves. 

The penalty may be increased based on the ridiculousness of the output (ie trying to move your opponent's pieces is worse than moving your own piece wrong?). 

The rewards will be modified based on the recency of the legal move that was chosen. I am sure that a3 is statistically a legal move far more often than Nh7, and I don't want the model to learn to just predict a3 over and over. So if it correctly chose a3 recently, it would get a smaller reward for correctly picking it again if it had another legal option, and it would get a higher reward for choosing a legal move that it hasn't chosen for a long time.

#### Supervised learning
A collection of chess positions will be curated (through random generation and/or randomly chosen positions in lichess chess960 games). These positions will then be evaluated by Stockfish 16 at high depth to ensure very high quality data. The Stockfish evaluation depth will be determined by the evaluation of the position at depth 20. The more even the position, the higher the depth, up to a depth of 50 or 60 for the most equal positions. This is to optimize the data labeling process, since close positions are more critical, and I don't want to waste additional compute power on clearly won/lost positions.

Since chess is solved for positions with 7 or fewer pieces, I will only use Stockfish to evaluate positions with greater than 7 pieces. Endgame databases will be used to get high quality labeled data for positions with 7 or fewer pieces.

#### Reinforcement learning
The model will also be trained using reinforcement learning through [self-play](https://en.wikipedia.org/wiki/Self-play), similar to AlphaZero or LeelaChessZero. I would like to employ this method both as a fine-tuning strategy after doing supervised learning, and as a standalone strategy to be used from scratch, then look at the results of both strategies.

I would also like to do self-play in a tournament structure rather than one-on-one matches. I believe this will reduce gaps in knowledge and result in a model that is the most well-rounded.

#### Adversarial model -> supervised learning
After using the previous methods (either supervised + self-play fine-tuning, or standalone self-play) I would like to train an adversarial network to learn the weaknesses of my model. This method was used to [defeat the superhuman Go AI, KataGo](https://arxiv.org/abs/2211.00241). Afterward, the American human, Kellin Pelrine, was able to learn this strategy to [defeat KataGo](https://arstechnica.com/information-technology/2023/02/man-beats-machine-at-go-in-human-victory-over-ai/). If my adversarial model is able to defeat the Minotaur model, I will then take positions from games that Minotaur lost and feed them into Stockfish 16 at high depth, and use that data to fine-tune the model with additional supervised learning.

This process will then be repeated.


## Reinforcement learning details
#### Evolutionary AI
After a base version of the AI has been trained, I will use an [evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) to train the model further. The model's weights represent a specific point in a parameter space. We define a hypersphere around the point, and instantiate a number of other models in that region of the parameter space. Those models will then compete in a large tournament, and the model with the best score will be kept. We then repeat the process of defining a hypersphere and choosing additional points within the resulting . 

#### Hyperspheres and Randomness
Choosing random points within a hypersphere is a bit more complicated than it sounds. The method of choosing can affect the distribution of the sample, similar to the [Bertrand Paradox](https://en.wikipedia.org/wiki/Bertrand_paradox_(probability)) where three different methods of choosing random lines in a circle result in three different distributions of points, such as biasing toward or against the center.

Luckily, [there are known ways to get a uniform distribution](https://mathworld.wolfram.com/HyperspherePointPicking.html). We can use Gaussian random variables to determine a random point on the surface of the hypersphere and then scale it inward toward the center. However, it needs to be noted that the [volume of a hypersphere](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) is concentrated toward the edges, so the scaling needs to be weighted to account for this.

Other methods, like randomly generating polar coordinates, or randomly choosing points on the surface of the hypersphere by doing random transformations to a unit vector before scaling inward, or much different methods, like choosing two random points on the surface, and choosing a random point on the connecting line, do not yield uniform distributions.

I would like to explore the effect that these different uniform vs non-uniform distributions have on the training process and the convergence of the model.



