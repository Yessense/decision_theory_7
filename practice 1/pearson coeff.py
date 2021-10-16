import numpy as np
from scipy.stats import pearsonr


def mean(arr: list) -> float:
    return sum(arr) / len(arr)


def cov(arr_x: list, arr_y: list) -> float:
    mean_x = mean(arr_x)
    mean_y = mean(arr_y)
    result = 0
    for i in range(len(arr_x)):
        result += (arr_x[i] - mean_x) * (arr_y[i] - mean_y)
    return result


def std_dev(arr: list) -> float:
    result = 0
    arr_mean = mean(arr)
    for i in range(len(arr)):
        result += (arr[i] - arr_mean) ** 2
    return result ** 0.5


def pearson_coeff(arr_x: list, arr_y: list) -> float:
    return cov(arr_x, arr_y) / (std_dev(arr_x) * std_dev(arr_y))


if __name__ == '__main__':
    matrix = np.random.rand(6, 6) * 10
    matrix = np.array([[4,5,4,4,3,3], [3,3,3,2,4,5]])

    my_l1 = [[0] * len(row) for row in matrix]
    sklearn_l1 = [[0] * len(row) for row in matrix]

    for i, r1 in enumerate(matrix):
        for j, r2 in enumerate(matrix):
            my_l1[i][j] = pearson_coeff(r1, r2)
            sklearn_l1[i][j] = pearsonr(r1, r2)[0]

    print(f'Matrix:\n{matrix}')

    print("My score")
    print(*my_l1, sep='\n')

    print("\nSklean scores:")
    print(*sklearn_l1, sep='\n')
