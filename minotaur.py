import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import threading
import random
import chess



# ##### ##### ##### ##### #####
# Architecture

class Minotaur(nn.Module):
    def __init__(self, mode="normal", pretrain_model=None):
        super(Minotaur, self).__init__()
        self.mode = mode
        self.input_layer = nn.Linear(64, 128)  # 64 squares input
        self.hidden_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(99)])

        if mode == "pretrain":
            self.num_predict = 10
            self.output_layer = nn.Linear(128, self.num_predict * 4)
        elif mode == "normal":
            self.num_predict = 1
            if pretrain_model is not None:
                pretrain_dict = {k: v for k, v in pretrain_model.state_dict().items() if "output_layer" not in k}
                self.load_state_dict(pretrain_dict, strict=False)
            self.output_layer = nn.Linear(128, self.num_predict * 4)
        else:
            # Handle an exception?
            self.num_predict = 1
            self.output_layer = nn.Linear(128, self.num_predict * 4)

    #
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = torch.tanh(self.output_layer(x))
        x = (x + 1) * 3.5 + 1  # Transforms: [-1, 1] -> [1, 8]
        return x.round().tolist()

# ##### ##### ##### ##### #####
# Pre-training

    #
    def pretrain(self, position_vector_df, batch_size, stop_event):
        is_fork = multiprocessing.get_start_method() == "fork"
        device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )
        lr = 3e-4
        max_grad_norm = 1.0


        # batch_size needs to become minibatch_size or something
        # probably doesn't need to be an argument either. Just define here with the rest of the parameters.

        sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
        num_epochs = 10  # optimization steps per batch of data collected
        clip_epsilon = (
            0.2  # clip value for PPO loss: see the equation in the intro for more context.
        )
        gamma = 0.99
        lmbda = 0.95
        entropy_eps = 1e-4



        # A loop to continuously process batches of positions
        for batch_start_index in range(0, len(position_vector_df) - batch_size, batch_size):
            # We check for stop condition here so that we don't stop mid-batch
            if not stop_event.is_set():
                # A list of positions, including those that result from the network's predictions
                positions_now_and_future = []

                # Loop through all positions in this batch
                for batch_progress_index in range(0, batch_size):
                    # This row of the dataframe is a FEN and position vector (network input)
                    this_row = position_vector_df.iloc[batch_start_index + batch_progress_index]

                    # Do I actually need this?
                    legal_moves = list(chess.Board(this_row["FEN"]).legal_moves)

                    # Append FEN column value to positions_now_and_future
                    positions_now_and_future.update(this_row["FEN"])

                    # Do a forward pass using this_row minus the FEN column
                    # Store output which is self.num_predict * 4 nodes
                    output = self.forward(this_row.drop(columns=["FEN"]))

                    # Convert output to a list of self.num_predict moves
                    output_moves = []
                    for i in range(0, len(output), 4):
                        # Use these 4 elements to get a move, then add it to output_moves

                    for move in output_moves:
                        if move in legal_moves:
                            # Do the move to the board
                            # Retrieve the corresponding FEN
                            # Add the FEN to positions_now_and_future
                        else:
                            # Break out of the loop. Don't add invalid positions to positions_now_and_future
                            break


                # At this point we've completed the current batch
                # Score this batch



            else:
                break

        # We've made predictions on one full batch


        pass


# ##### ##### ##### ##### #####
# Supervised learning

# The model will be trained via supervised learning on chess960 positions analyzed at high depth.
# It will also be trained on endgame databases, and forced checkmate sequences.
# The latter two represent perfect quality data. I'm interested in how each of these sources impacts performance.


# ##### ##### ##### ##### #####
# Reinforcement learning


# ##### ##### ##### ##### #####
# Adversarial learning

# I would then like to train an adversarial network against it to attempt to find gaps in its knowledge.
# Then I will feed positions from games it loses against the adversarial network into Stockfish at very high
# depth and repeat the process.


# ##### ##### ##### ##### #####
# Program Body

if __name__ == "__main__":
    stop_event = threading.Event()

















