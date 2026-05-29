import numpy as np
import pandas as pd
from pathlib import Path


def BifDiagram(W_file="normalized_matrix_4cluster.npy"):

    root = Path(__file__).resolve().parent.parent
    W = np.load(root / "data" / W_file)
    Npop = W.shape[0]

    # ================ LOAD DATA =================

    floquet_data = np.load(root / "scripts" / f'FloquetBifDiagram_Npop={Npop}.npz')
    eq_data = np.load(root / "scripts" / f'EqPointsBifDiagram_Npop={Npop}.npz')

    # Grids
    vector_Iext_e = floquet_data['vector_Iext_e']
    vector_eps = floquet_data['vector_eps']

    # Consistency 
    assert np.allclose(vector_Iext_e, eq_data['vector_Iext_e'])
    assert np.allclose(vector_eps, eq_data['vector_eps'])

    # Floquet data
    dataFloquetReal = floquet_data['dataFloquetReal']
    dataStatus = floquet_data['dataStatus']
    numPositiveFloquet = floquet_data['numPositive']
    ICs_PO = floquet_data['ICs']
    Ts = floquet_data['Ts']

    # EqPoints data
    dataVapsReal = eq_data['dataVapsReal']
    numPositiveEq = eq_data['numPositive']
    ICs_EP = eq_data['ICs']

    Nvariables = ICs_PO.shape[2]

    # ================= MERGE =================

    rows = []
    dataStatus = dataStatus.astype(int)

    # Unified arrays (for npz)
    merged_exponents = np.zeros_like(dataFloquetReal)
    merged_numPositive = np.zeros_like(numPositiveFloquet)
    merged_ICs = np.zeros_like(ICs_PO)
    merged_Ts = np.zeros_like(Ts)

    for i in range(len(vector_Iext_e)):
        for j in range(len(vector_eps)):

            status = dataStatus[i, j]

            row = [
                vector_Iext_e[i],
                vector_eps[j],
                status,
            ]

            # ---------------- CASE 1: FIXED POINTS ----------------
            if status == 0:
                IC = ICs_EP[i, j]
                exponents = dataVapsReal[i, j]
                numPos = numPositiveEq[i, j]
                T = np.nan

            # ---------------- CASE 2: PERIODIC ORBITS ----------------
            elif status in [2, 4, 5]:
                IC = ICs_PO[i, j]
                exponents = dataFloquetReal[i, j]
                numPos = numPositiveFloquet[i, j]
                T = Ts[i, j]

            # ---------------- CASE 3: CHAOS ----------------
            else:
                IC = np.nan * np.ones(Nvariables)
                exponents = np.nan * np.ones(Npop)
                numPos = np.nan
                T = np.nan

            # Save into merged arrays
            merged_exponents[i, j] = exponents
            merged_numPositive[i, j] = numPos
            merged_ICs[i, j] = IC
            merged_Ts[i, j] = T

            # Build row
            row.append(T)

            # Expand ICs
            if np.isnan(IC).all():
                row.extend([np.nan]*Nvariables)
            else:
                row.extend(IC.tolist())

            # Exponents
            row.extend(exponents.tolist())

            # Num positive
            row.append(numPos)

            rows.append(row)

    # ================= TABLE =================

    columns = (
        ["Iext_e", "eps", "status", "T"] +
        [f"x{k}" for k in range(Nvariables)] +
        [f"lambda_{k}" for k in range(Npop)] +
        ["numPositive"]
    )

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(
        root / "scripts" / f"BifDiagram_Npop={Npop}.txt",
        sep=" ",
        index=False,
        na_rep="NaN"
    )

    # ================= SAVE NPZ =================

    np.savez(
        root / "scripts" / f'BifDiagram_Npop={Npop}.npz',
        vector_Iext_e=vector_Iext_e,
        vector_eps=vector_eps,
        dataStatus=dataStatus,
        exponents=merged_exponents,
        numPositive=merged_numPositive,
        ICs=merged_ICs,
        Ts=merged_Ts
    )

    return df
