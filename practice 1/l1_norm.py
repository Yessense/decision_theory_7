import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

def my_manhattan_distance(x, y):
    assert len(x) == len(y)

    sum = 0
    for el1, el2 in zip(x, y):
        sum += abs(el1 - el2)

    return sum

if __name__ == '__main__':
    matrix = np.random.rand(6,6) * 10

    my_l1 = [[0] * len(row) for row in matrix]
    sklearn_l1 = [[0] * len(row) for row in matrix]


    for i, r1 in enumerate(matrix):
        for j, r2 in enumerate(matrix):
            my_l1[i][j] = my_manhattan_distance(r1, r2)
            r1_np = np.array([r1])
            r2_np = np.array([r2])
            sklearn_l1[i][j] = manhattan_distances(r1_np, r2_np)[0,0]

    print(f'Matrix:\n{matrix}')

    print("My score")
    print(*my_l1, sep='\n')

    print("\nSklean scores:")
    print(*sklearn_l1, sep='\n')

