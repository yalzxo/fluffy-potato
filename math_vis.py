import numpy as np
import pandas as pd

df = pd.read_csv("/Users/mmadando/Downloads/Netflix_stock_data.csv")
print("Columns in dataset:", df.columns)

print("First 5 rows:\n", df.head())
df = df.drop(columns=["Date"])
print("\nColumns after dropping Date:\n", df.columns)

target_column = "Close"

A = df.drop(columns=[target_column]).values
b = df[target_column].values

print("\nFeature matrix A shape:", A.shape)
print("Target vector b shape:", b.shape)

print("\n--- QR Decomposition ---")
Q, R = np.linalg.qr(A)
print("Q shape:", Q.shape)
print("R shape:", R.shape)

A_qr_reconstructed = Q @ R
error_qr = np.linalg.norm(A - A_qr_reconstructed)
print("QR Reconstruction error:", error_qr)


print("\n--- Singular Value Decomposition (SVD) ---")
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S = np.diag(s)
print("U shape:", U.shape)
print("Singular values:", s)
print("V^T shape:", Vt.shape)

A_svd_reconstructed = U @ S @ Vt
error_svd = np.linalg.norm(A - A_svd_reconstructed)
print("SVD Reconstruction error:", error_svd)

print("\n--- Least Squares Approximation ---")
x_ls, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
print("Least squares solution (x):\n", x_ls)
print("Residuals:", residuals)

b_pred = A @ x_ls
print("\nFirst 10 predicted Close values:\n", b_pred[:10])
print("First 10 actual Close values:\n", b[:10])

print("\n--- Eigenvalues of A^T A ---")
M = A.T @ A
eigvals, eigvecs = np.linalg.eig(M)
print("Eigenvalues:\n", eigvals)

U, s, Vt = np.linalg.svd(A, full_matrices=False)
eigvals_from_svd = s**2
print("\nEigenvalues from SVD (singular values squared):\n", eigvals_from_svd)

rank = np.linalg.matrix_rank(A)
n_features = A.shape[1]
print("Rank:", rank)
print("Number of columns:", n_features)

if rank < n_features:
    print("Columns are linearly dependent.")
else:
    print("Columns are linearly independent.")

Q, R = np.linalg.qr(A)
print("Orthonormal basis (Q):\n", Q)
