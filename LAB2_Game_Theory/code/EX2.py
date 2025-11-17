import nashpy as nash
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Shows problem")
# Define the payoff matrices
    A = np.array([[4, 0], [0, 2]])
    B = np.array([[2, 0], [0, 4]])
# Create the game
    game = nash.Game(A, B)
    equilibrium = list(game.support_enumeration())
    print(equilibrium)