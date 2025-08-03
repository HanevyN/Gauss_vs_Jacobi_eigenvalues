import numpy as np


def rq_decomposition(A):
    A_flipped = np.flipud(A).T
    Q, R = np.linalg.qr(A_flipped)
    R = np.flipud(R.T)
    Q = Q.T[:, ::-1]
    return R, Q


def generate_test_case(repeat_value=2.0):
    A = np.random.rand(3, 3)
    R, Q = rq_decomposition(A)

    B = np.eye(3)
    B[1, 1] = repeat_value

    A_symmetric = Q.T @ B @ Q
    return A_symmetric


def export_to_header(matrix_list, filename="test_matrices.h"):
    with open(filename, "w") as f:
        f.write("#pragma once\n\n")
        f.write("constexpr int NUM_TEST_CASES = {};".format(len(matrix_list)))
        f.write("constexpr double TEST_MATRICES[NUM_TEST_CASES][3][3] = {\n")
        for A in matrix_list:
            f.write("    { {" + ", ".join(f"{A[0, i]:.16f}" for i in range(3)) + "},\n")
            f.write("      {" + ", ".join(f"{A[1, i]:.16f}" for i in range(3)) + "},\n")
            f.write(
                "      {" + ", ".join(f"{A[2, i]:.16f}" for i in range(3)) + "} },\n"
            )
        f.write("};\n")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    test_matrices = [generate_test_case(repeat_value=2.0) for _ in range(10)]
    export_to_header(test_matrices)
    print("Exported 10 test matrices to test_matrices.h")
