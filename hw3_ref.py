import numpy as np
from Bio import SeqIO

def parse_score_matrix(file_path):
    """Parse substitution matrix (PAM100 or PAM250)."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
    headers = lines[0].split()
    matrix = {}
    for line in lines[1:]:
        parts = line.split()
        row = parts[0]
        scores = list(map(int, parts[1:]))
        matrix[row] = dict(zip(headers, scores))
    return matrix

def alignment(input_path, score_path, output_path, aln_type, gap_score):
    """Perform alignment using M, Ix, and Iy matrices."""
    # Parse substitution matrix
    score_matrix = parse_score_matrix(score_path)

    # Parse sequences from input FASTA file
    records = list(SeqIO.parse(input_path, "fasta"))
    seq1, seq2 = str(records[0].seq), str(records[1].seq)
    seq1_name, seq2_name = records[0].id, records[1].id

    len1, len2 = len(seq1), len(seq2)
    M = np.zeros((len1 + 1, len2 + 1), dtype=float)  # Use float to handle -inf
    Ix = np.zeros((len1 + 1, len2 + 1), dtype=float)
    Iy = np.zeros((len1 + 1, len2 + 1), dtype=float)

    # Initialize matrices
    M.fill(float('-inf'))
    Ix.fill(float('-inf'))
    Iy.fill(float('-inf'))

    if aln_type == "global":
        M[0, 0] = 0
        for i in range(1, len1 + 1):
            Ix[i, 0] = gap_score * i
        for j in range(1, len2 + 1):
            Iy[0, j] = gap_score * j
    elif aln_type == "local":
        M[:, :] = 0
        Ix[:, :] = 0
        Iy[:, :] = 0

    # Fill the DP matrices
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Scores for M(i, j)
            match_score = score_matrix[seq1[i - 1]].get(seq2[j - 1], gap_score)
            M[i, j] = max(
                M[i - 1, j - 1] + match_score,
                Ix[i - 1, j - 1] + match_score,
                Iy[i - 1, j - 1] + match_score,
            )

            # Scores for Ix(i, j)
            Ix[i, j] = max(
                M[i - 1, j] + gap_score,
                Ix[i - 1, j] + gap_score,
            )

            # Scores for Iy(i, j)
            Iy[i, j] = max(
                M[i, j - 1] + gap_score,
                Iy[i, j - 1] + gap_score,
            )

            if aln_type == "local":
                M[i, j] = max(M[i, j], 0)
                Ix[i, j] = max(Ix[i, j], 0)
                Iy[i, j] = max(Iy[i, j], 0)

    # Traceback for local alignment
    if aln_type == "local":
        max_positions = []
        max_score = max(M.max(), Ix.max(), Iy.max())
        if max_score == M.max():
            max_positions = list(zip(*np.where(M == max_score)))
        elif max_score == Ix.max():
            max_positions = list(zip(*np.where(Ix == max_score)))
        else:
            max_positions = list(zip(*np.where(Iy == max_score)))

        best_alignment = None
        best_length = 0
        for pos in max_positions:
            aligned_seq1, aligned_seq2 = "", ""
            i, j = pos
            while i > 0 and j > 0:
                # Allow traceback to include zero scores
                if M[i, j] >= Ix[i, j] and M[i, j] >= Iy[i, j] and M[i, j] >= 0:
                    aligned_seq1 = seq1[i - 1] + aligned_seq1
                    aligned_seq2 = seq2[j - 1] + aligned_seq2
                    i, j = i - 1, j - 1
                elif Ix[i, j] >= M[i, j] and Ix[i, j] >= Iy[i, j] and Ix[i, j] >= 0:
                    aligned_seq1 = seq1[i - 1] + aligned_seq1
                    aligned_seq2 = "-" + aligned_seq2
                    i -= 1
                elif Iy[i, j] >= M[i, j] and Iy[i, j] >= Ix[i, j] and Iy[i, j] >= 0:
                    aligned_seq1 = "-" + aligned_seq1
                    aligned_seq2 = seq2[j - 1] + aligned_seq2
                    j -= 1
                else:
                    break  # Stop traceback if all scores are < 0

            if len(aligned_seq1) > best_length:
                best_alignment = (aligned_seq1, aligned_seq2)
                best_length = len(aligned_seq1)

        aligned_seq1, aligned_seq2 = best_alignment

    # Add unaligned prefixes for global alignment
    elif aln_type == "global":
        aligned_seq1, aligned_seq2 = "", ""
        i, j = len1, len2
        while i > 0 or j > 0:
            if i > 0 and j > 0 and M[i, j] >= max(Ix[i, j], Iy[i, j]):
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                i, j = i - 1, j - 1
            elif i > 0 and (j == 0 or Ix[i, j] >= Iy[i, j]):
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                j -= 1

    # Save to output FASTA
    with open(output_path, 'w') as f:
        f.write(f">{seq1_name}\n{aligned_seq1}\n")
        f.write(f">{seq2_name}\n{aligned_seq2}\n")

# Example usage
# alignment("test_global.fasta", "pam250.txt", "result_global.fasta", "global", -10)
# alignment("test_local.fasta", "pam100.txt", "result_local.fasta", "local", -10)
