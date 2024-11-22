# Scoring: use these and make the king like 30 https://www.chess.com/terms/chess-piece-value#Chesspiecevals
# Gain points for taking pieces, lose points for losing pieces
# or maybe 10x those and give points for each move without losing the game
#
# Action Space:
# so there are 64 squares on the board and 32 pieces, well, only 16 that you can move, too much to enumerate
# but each piece would have a finite set of moves, eg pawn can go forward one, forward 2 (sometimes), foreward and left,
# forward and right, en passant (https://en.wikipedia.org/wiki/En_passant). So how many moves are there?
# Pawns: 5 * 8 = 40
# Knights: 8 * 2 = 16
# Bishops: 28 * 2 = 56
# Rooks: 28 * 2 = 56
# Queen: 56
# King: 8
# Total: 232
# That's a lot of potentially possible actions. But what if we separate it out so there are 16 output nodes for the 16
# pieces and an additional 56 for the up-to 56 possible moves. Then we have to teach it to only do legal moves.
# Picking the highest legal operation might work. Like select max from the first 16 then select max of a subset of the
# 56 moves that are legal for that piece.
#
# Inputs: so there are 64 squares, 32 pieces that we need to comminicate the positions of. If we set up the nodes so
# that node position indicates piece, and we have 2 values from 0 to 8 discreet to represent the x and y coordinates.
# Make 0,0 mean off board. 32 input nodes, 72 output nodes... this thing could take some time to train.
