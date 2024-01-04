import chess
import chess.engine


# Read in generated/collected board positions
# Feed those positions into Stockfish. Analyze to a depth of 20-50+ depending on how much of an advantage.
# (Positions where one side is overwhelmingly winning analyzed to lower depth, even positions evaluated to higher.)
# Only collect the top next move, and use it to label the board position.

# I think this program should maintain multiple files.
# If you evaluated some positions to depth 20, some to 30, 40, etc... Those should be kept separate from each other
# so, you need to decide how many categories to do based on the evaluation, and maintain that many files.








