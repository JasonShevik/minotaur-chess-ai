import tensorflow


# Since this Chess AI only looks at the next move, and does not do search, then it will be a classifier AI.
# Have two output nodes, one for the starting square, and one for the destination square.
# Nodes can choose any one of 64 squares, plus two additional squares represent castling. (65,65), (66,66) mean castle.


# A board position should look like this:
# The pieces on the board
# [[x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
# Right to castle
#  [x,x],
# Right to en passant
#  [x,x,x,x,x,x,x,x]]


# ### Training process ###
# These 74 inputs will be used to create numerous polynomial features, as well as fourier features,
# then fed into a relatively large neural network.

# The network will be trained via supervised learning on very high quality single-move chess puzzles created
# by giving generated chess positions to stockfish to be evaluated at very high depth, as well as positions
# from endgame databases where correct moves are proven.
# This will bring the model to a base level. I'm interested to see what this level will be.

# The bot will then be fine-tuned using reinforcement learning through self play.
# Rather than having two versions play each other and picking the winner, this fine-tuning will be to have
# a small tournament with multiple versions of the AI and choosing the winner. This method should
# reduce gaps in knowledge, and ensure that the model converges to a well-rounded style that is resistant
# to exploitation.

# I would then like to train an adversarial network against it to attempt to find gaps in its knowledge.
# Then I will feed positions from games it loses against the adversarial network into Stockfish at very high
# depth and repeat the process.








