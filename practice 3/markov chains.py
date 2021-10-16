import numpy as np


def resolve_markov_chain(matrix):
    print(f'Matrix:\n{matrix}')
    size = len(matrix)

    add_matrix = np.logical_xor(np.eye(size), np.ones_like(matrix))
    print(f'Matrix to add:\n{add_matrix}')

    X = (matrix + add_matrix).T
    y = np.ones(size)

    print(f'Equation matrix:\n{X}\n{y}')

    result = np.linalg.solve(X, y)
    print(f'Result probabilities:\n{result}')

def resolve_continuous_time_markov_chain(matrix):
    print(f'Matrix:\n{matrix}')
    size = len(matrix)

    matrix[-1] = np.ones(size)
    print(f'Matrix without last row:\n{matrix}')

    X = matrix
    y = np.zeros(size)
    y[-1] = 1
    print(f'Equation matrix:\n{X}\n{y}')

    result = np.linalg.solve(X,y)
    print(f'Result probabilities:\n{result}')



if __name__ == '__main__':
    matrix = np.array([[0.7, 0.2, 0.1],
                       [0.8, 0.1, 0.1],
                       [0.8, 0.05, 0.15]])
    resolve_markov_chain(matrix)

    matrix2 = np.array([[-3, 2, 3, 0],
                        [1, -4, 0, 3],
                        [2, 0, -4, 2],
                        [0, 2, 1, -5]])
    resolve_continuous_time_markov_chain(matrix2)