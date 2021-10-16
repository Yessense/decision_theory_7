from sklearn.metrics import jaccard_score
import numpy as np


def my_jaccard_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    intersection = 0
    union = 0

    for elem_1, elem_2 in zip(y_true, y_pred):
        if elem_1 and elem_2:
            intersection += 1
            union += 1
        elif elem_1 or elem_2:
            union += 1

    if not union:
        return 0

    return intersection / union


if __name__ == '__main__':
    matrix = np.random.randint(0, 2, (6, 6))

    print(f'Matrix:\n{matrix}')

    my_score = [[0] * len(row) for row in matrix]
    sklearn_score = [[0] * len(row) for row in matrix]

    for i, r1 in enumerate(matrix):
        for j, r2 in enumerate(matrix):
            my_score[i][j] = my_jaccard_score(r1, r2)
            sklearn_score[i][j] = jaccard_score(r1, r2)

    print("My score")
    print(*my_score, sep='\n')

    print("\nSklean scores:")
    print(*sklearn_score, sep='\n')
