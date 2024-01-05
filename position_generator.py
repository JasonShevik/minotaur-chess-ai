import keyboard


# Generation rules:
# 1. No pawns on the first or last rank.
# 2. Only one king per color.
# 3. No more than 8 pawns of a single color.
# 4. a. No more than 3 pawns of a single color on a single file.
# 4. b. If more than 1 of single color pawn on a file, restrict number of pawns in adjacent files.
# 5. No more than 3 of a single minor piece for each color.
# 6. Define a threshold for the maximum material advantage over opponent.
# 7. Define a flag to determining if castling is allowed in a position.
# 8. Define a flag to determine if en passant is allowed in a position (and which pawn(s) can do it).
# 9. A "pass" of this program is when it generates board positions with 32 pieces all the way down to 7.


# ### Main functions ###

def generate_random_pass(piece_permutations_pass):
    """
    One "pass" is creating a position for each number of pieces left on the board from all 32 down to 8.
    :param piece_permutations_pass:
    :return: A list of 25 generated chess board configurations with 8 to 32 pieces on the board
    """
    this_pass = []
    for num_pieces in range(32, 8, -1):
        this_pass.append(generate_board(piece_permutations_pass[num_pieces]))
    return this_pass


def generate_standard_openings_pass():
    """

    :return:
    """
    return []


def generate_960_openings_pass():
    """

    :return:
    """
    return []


# ### Helper functions ###

def get_piece_permutations(num_pieces_get, max_difference_get):
    """

    :param num_pieces_get:
    :param max_difference_get:
    :return:
    """
    all_piece_permutations = []
    piece_values = [1, 3, 3, 5, 9]  # [Pawn, Knight, Bishop, Rook, Queen]
    piece_maxes = [8, 3, 3, 3, 2]   # Maximum number of each piece allowed on a board

    # Some kind of loop here
    set_one = [0, 0, 0, 0, 0]
    set_two = [0, 0, 0, 0, 0]

    # sum(set_one) + sum(set_two) == num_pieces_get
    # abs(sum(set_one) - sum(set_two)) == max_difference_get

    all_piece_permutations.append([set_one, set_two])

    return all_piece_permutations


def generate_board(these_permutations):
    """

    :param these_permutations:
    :return:
    """
    board_config = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]

    # I should place the pawns first

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

    return board_config


def export_positions(positions_list, board_type):
    """

    :param positions_list:
    :param board_type: A flag to indicate whether these are "random" positions, "standard openings", or "960 openings"
    :return:
    """
    return False


# ### Generation ###

# Tuning variables
max_difference = 3  # Maximum material imbalance allowed in a generated position.
generation_type = "random"  # Generate "random" positions, "standard openings", or "960 openings"

# Other variables
piece_permutations_dict = {}
generated_boards = []


if generation_type is "random":
    for num_pieces_pass in range(32, 8, -1):
        piece_permutations_dict[num_pieces_pass] = get_piece_permutations(num_pieces_pass, max_difference)

    # Consider making the whole next part its own function. Then I can call that in all three cases instead of copying

    while True:
        generated_boards.append(generate_random_pass(piece_permutations_dict))

        if keyboard.is_pressed('q'):  # Do I need a try-catch around this?
            export_positions(generated_boards, generation_type)
            exit(0)


elif generation_type is "standard openings":
    while True:
        generated_boards.append(generate_standard_openings_pass())

        if keyboard.is_pressed('q'):
            export_positions(generated_boards, generation_type)
            exit(0)
elif generation_type is "960 openings":
    while True:
        generated_boards.append(generate_960_openings_pass())

        if keyboard.is_pressed('q'):
            export_positions(generated_boards, generation_type)
            exit(0)

















