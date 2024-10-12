import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import threading
import random
import chess
from typing import Dict, List


# ##### ##### ##### ##### #####
# Class
class Minotaur(torch.nn.Module):
    def __init__(self, in_channels: int = 13, hidden_channels: int = 16, out_channels: int = 4, num_heads: int = 4, dropout: float = 0.6):
        super(Minotaur, self).__init__()

        # Prepare some values
        self.dropout: float = dropout
        num_gat_layers: int = 3
        num_lin_layers: int = 10
        gat_hidden_num: int = hidden_channels
        linear_hidden_num: int = 256
        num_edge_types: int = 6

        # Initialize the layer_list, and add the first GATConv layer
        self.gat_layers: torch.nn.ModuleList = torch.nn.ModuleList([GATConv(in_channels, gat_hidden_num, heads=num_heads, edge_dim=num_edge_types, dropout=dropout)])
        # Add all of the subsequent GatConv layers
        for _ in range(num_gat_layers - 1):
            gat_hidden_num = gat_hidden_num * num_heads
            self.gat_layers.append(GATConv(gat_hidden_num, gat_hidden_num, heads=num_heads, edge_dim=num_edge_types, dropout=dropout))

        # Initialize the linear_list, and add the first Linear layer
        self.linear_layers: torch.nn.ModuleList = torch.nn.ModuleList([torch.nn.Linear(gat_hidden_num, linear_hidden_num)])
        # Add all of the subsequent Linear layers
        for _ in range(num_lin_layers - 1):
            self.linear_layers.append(torch.nn.Linear(linear_hidden_num, linear_hidden_num))

        # PReLU activations for each layer
        self.gat_prelu: torch.nn.ModuleList = torch.nn.ModuleList([torch.nn.PReLU() for _ in range(len(self.gat_layers))])
        self.linear_prelu: torch.nn.ModuleList = torch.nn.ModuleList([torch.nn.PReLU() for _ in range(len(self.linear_layers))])

        # Output layer
        self.output = torch.nn.Linear(linear_hidden_num, out_channels)


    def forward(self, x, edge_index, edge_type):
        # Apply GAT layers
        i: int
        layer: nn.Module
        for i, layer in enumerate(self.gat_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index, edge_type)
            x = self.gat_prelu[i](x)

        # Apply Linear layers
        i: int
        layer: nn.Module
        for i, layer in enumerate(self.linear_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            x = self.linear_prelu[i](x)

        # Output layer
        x = torch.tanh(self.output(x))
        x = (x + 1) * 3.5 + 1  # Transforms: [-1, 1] -> [1, 8]

        return x.round().tolist()


    # -------------------------------------------------------------
    # This train method will probably have to be totally rewritten
    """
    def train(self, position_vector_df, batch_size, stop_event):
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
    """

    def save(self) -> None:
        pass


# ##### ##### ##### ##### #####
# Functions

# A function that runs a monte carlo simulation on a list of chess positions
# randomly choosing a move for each position sim_num times
def _monte_carlo_sim(position_vector_batch_df: pd.DataFrame, sim_num: int) -> Dict[str, int]:
    """

    :param position_vector_batch_df:
    :param sim_num:
    :return:
    """
    # A dictionary to hold a count of randomly chosen / simulated moves
    sim_dict: Dict[str, int] = {}
    # A set to keep track of all legal moves that appear
    legal_set: set = set()

    # Simulate sim_num times
    for _ in range(0, sim_num):
        # Loop through each position
        for i in range(0, len(position_vector_batch_df)):
            # Get the current row
            this_row: pd.DataFrame = position_vector_batch_df.iloc[i]

            # Get the list of legal moves using the chess library and the FEN string, then convert SAN to UCI
            legal_moves: List[str] = get_legal_moves_strings(this_row["FEN"])

            # Make sure that these moves are in the legal_set
            legal_set.update(legal_moves)

            # Randomly choose a legal move
            sim_move: str = random.choice(legal_moves)
            # Increment or initialize the simulated move in sim_dict
            if sim_move in sim_dict:
                sim_dict[sim_move] += 1
            else:
                sim_dict[sim_move] = 1

    # Normalize the results
    move: str
    for move in sim_dict:
        sim_dict[move] /= sim_num

    # Make sure that the sim_dict contains all legal moves, even if it never picked some by chance
    for move in legal_set:
        if move not in sim_dict:
            sim_dict[move] = 0

    return sim_dict


#
def _score_random(position_vector_batch_df: pd.DataFrame, choices_dict: Dict[str, List[str]], illegal_move_penalty_factor: float = 1) -> float:
    """

    :param position_vector_batch_df:
    :param choices_dict:
    :param illegal_move_penalty_factor:
    :return:
    """

    total_penalty: float = 0
    total_reward: float = 0

    # Start by calculating the number of illegal moves and summing a penalty
    total_penalty += (illegal_move_penalty_factor *
                      position_vector_batch_df['FEN'].apply(lambda x: _get_num_invalid_moves(x, choices_dict[x])).sum())


    sim_dict: Dict[str, int] = _monte_carlo_sim(position_vector_batch_df, 1000)
    move: int
    for move in sim_dict:
        # Calculate the sum of absolute difference in values for choices_dict and sim_dict
        total_penalty +=

    return total_reward - total_penalty


# A function to return the number of invalid moves predicted by the network
def _get_num_invalid_moves(fen: str, moves: List[str]) -> int:
    """

    :param fen:
    :param moves:
    :return:
    """
    # If we've reached the base case with no moves left, then all moves were valid
    if not moves:
        return 0
    # If the first move in the list is legal
    if moves[0] in get_legal_moves_strings(fen):
        # Recursively call this function using the fen that results from that move and return the list shortened by 1
        return _get_num_invalid_moves(chess.Board(fen).push(chess.Move.from_uci(moves[0])).fen(), moves[1:])
    else:
        # If the move was illegal, then all moves left in the list are illegal
        return len(moves)


#
def get_legal_moves_strings(fen: str) -> List[str]:
    return [board.parse_san(move).uci() for move in list(chess.Board(fen).legal_moves)]





