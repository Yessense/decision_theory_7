import numpy as np
from scipy import linalg
from sklearn.preprocessing import normalize

np.set_printoptions(suppress=True)


def svd(X: np.ndarray):
    print(f"Исходная матрица:\n{X}\n")

    A = X.T @ X
    print(f'X * X.T:\n{A}\n')

    eigh_val, eigh_vec = np.linalg.eigh(A)

    print(f'Собственные значения:\n {eigh_val}\n')
    print(f'Собственные векторы:\n {eigh_vec}\n')

    V = eigh_vec[:]
    print(f'V matrix:\n{V}\n')

    s = np.diag(eigh_val)
    print(f'S matrix:\n{s}\n')

    U = X @ V / eigh_val

    print(f"U matrix:\n{U}\n")

    return U, s, V




if __name__ == '__main__':
    # X = np.random.randint(0, 2, (3, 4))

    A = np.array([[1, 1, 1, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1]])

    U, s, Vh = linalg.svd(A, full_matrices=False)
    print("New")
    print(U)
    print(s)
    print(Vh)

    U_1, s_1, V_1 = svd(A)
    print(U_1)
    print(s_1)
    print(V_1)
    s = np.diag(s)

    print(U_1 @ s_1 @ V_1.T)