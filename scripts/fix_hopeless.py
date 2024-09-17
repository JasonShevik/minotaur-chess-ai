# This code was for fixing labeled positions after I changed methodologies.
# There used to be a 'hopeless' mechanic such that if the score was too high, the position was deemed 'hopeless'
# And the engine wouldn't analyze further. This was to save time and compute resources.
# I ultimately decided that I would prefer that all positions be analyzed to at least depth 20 or forced checkmate.
def analyze_hopeless(engine, analysis_path, depth_score_breaks):
    # This empty set will hold all of the analyzed positions to print later
    file_rows = set()
    # Open the results file that we're re-analyzing
    with open(analysis_path, "r") as analysis_file:
        # Skip the header row
        next(analysis_file)
        foo = 1
        # Go through each line in the file
        for line in analysis_file:
            print(foo)
            foo += 1
            # Split the row into the different columns so that we can look at them individually
            row_elements = line.split(",")
            # If this row was analyzed below the minimum depth and wasn't a forced checkmate
            if int(row_elements[2]) < depth_score_breaks[0][0] and "#" not in row_elements[3]:
                # Set the board object for this position
                board = chess.Board(row_elements[0])
                # Set the index that we use with depth_score_breaks to determine depth based on a score threshold
                threshold_index = 0
                # Begin the analysis of this position
                with engine.analysis(board) as analysis:
                    # Number of nodes explored is 0 at the start of the analysis
                    num_nodes_last = 0
                    # Analyze continuously, depth by depth, until we meet a break condition
                    for info in analysis:
                        # Get the current depth
                        depth = info.get("depth")
                        if depth is None:
                            continue
                        # Get the current score
                        score = info.get("score")
                        # Score can be None when looking at sidelines - Skip those iterations
                        if score is None:
                            continue
                        score = str(score.pov(True))
                        # If we discovered a checkmate before we realized it's hopeless
                        if "#" in score:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Add to the file rows set
                            file_rows.add(f"{row_elements[0]},{best_move},{depth},{str(info.get('score').pov(True))}")
                            break
                        # Retrieve the absolute value of the score
                        score = abs(int(score))
                        num_nodes = info.get("nodes")
                        if num_nodes_last >= num_nodes:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Add to the file rows set
                            file_rows.add(f"{row_elements[0]},{best_move},{depth},{str(info.get('score').pov(True))}")
                            break
                        num_nodes_last = num_nodes
                        # If we've reached a depth milestone specified by depth_score_breaks
                        if depth == depth_score_breaks[threshold_index][0]:
                            # If the score is greater than allowed by depth_score_breaks for this depth
                            if score > depth_score_breaks[threshold_index][1]:
                                # Get the first move of the principle variation
                                best_move = info.get("pv")[0]
                                # Add to the file rows set
                                file_rows.add(f"{row_elements[0]},{best_move},{depth},{str(info.get('score').pov(True))}")
                                break
                            else:
                                # The evaluation is close enough that we want to analyze deeper
                                threshold_index += 1
            # If this row was analyzed properly, just add it to the list
            else:
                file_rows.add(line)
        # Print file_rows to a new file
        with open("results no hopeless.csv", "w") as output_file:
            output_file.write("Position,Move,Depth,Score\n")
            for row in file_rows:
                output_file.write(row.rstrip() + "\n")
# If you're redoing the analysis from hopeless mechanic
doing_hopeless = False
if doing_hopeless:
    which = "stockfish"
    if which == "leela":
        the_breaks = leela_breaks
        the_config = leela_config
    elif which == "stockfish":
        the_breaks = stockfish_breaks
        the_config = stockfish_config
    else:
        exit(0)
    hopeless_engine = hf.initialize_engine(which, the_config)
    analyze_hopeless(hopeless_engine, f"output-{which}/results.csv", the_breaks)
    hopeless_engine.quit()
    exit(0)