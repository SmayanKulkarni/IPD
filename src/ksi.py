import numpy as np

def calculate_angle_3d(a, b, c):
    v1 = a - b
    v2 = c - b
    denominator = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cosine_angle = np.dot(v1, v2) / denominator
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_ksi_features(landmarks_frame):
    p = lambda i: landmarks_frame[i]
    L_S, L_E, L_W = 11, 13, 15
    R_S, R_E, R_W = 12, 14, 16
    L_H, L_K, L_A = 23, 25, 27
    R_H, R_K, R_A = 24, 26, 28

    mid_shoulder = (p(L_S) + p(R_S)) / 2
    mid_hip = (p(L_H) + p(R_H)) / 2
    torso_len = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6

    return np.array([
        calculate_angle_3d(p(L_S), p(L_E), p(L_W)),
        calculate_angle_3d(p(R_S), p(R_E), p(R_W)),
        calculate_angle_3d(mid_hip, p(L_S), p(L_E)),
        calculate_angle_3d(mid_hip, p(R_S), p(R_E)),
        calculate_angle_3d(p(L_H), p(L_K), p(L_A)),
        calculate_angle_3d(p(R_H), p(R_K), p(R_A)),
        calculate_angle_3d(p(L_K), p(L_H), mid_shoulder),
        calculate_angle_3d(p(R_K), p(R_H), mid_shoulder),
        np.linalg.norm(p(L_S) - p(L_W)) / torso_len,
        np.linalg.norm(p(R_S) - p(R_W)) / torso_len,
        np.linalg.norm(p(L_H) - p(L_A)) / torso_len,
        np.linalg.norm(p(R_H) - p(R_A)) / torso_len
    ])

def dynamic_time_warping(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        steps = [dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]]
        best = np.argmin(steps)
        i -= (best != 1)
        j -= (best != 0)
    path.reverse()
    idx1, idx2 = zip(*path)
    return seq1[list(idx1)], seq2[list(idx2)]

def calculate_ksi(expert_seq, user_seq, weights):
    if not len(expert_seq) or not len(user_seq): return {'ksi_total': 0.0}
    expert_aligned, user_aligned = dynamic_time_warping(expert_seq, user_seq)
    
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    s_pose = np.mean([cosine_sim(expert_aligned[i], user_aligned[i]) for i in range(len(expert_aligned))])
    
    vel_exp = np.diff(expert_aligned, axis=0, prepend=expert_aligned[0:1])
    vel_user = np.diff(user_aligned, axis=0, prepend=user_aligned[0:1])
    s_velocity = np.mean([cosine_sim(vel_exp[i], vel_user[i]) * np.exp(-0.1 * (np.linalg.norm(vel_exp[i]) - np.linalg.norm(vel_user[i]))**2) for i in range(len(vel_exp))])

    acc_exp = np.diff(vel_exp, axis=0, prepend=vel_exp[0:1])
    acc_user = np.diff(vel_user, axis=0, prepend=user_aligned[0:1])
    s_accel = np.mean([np.exp(-0.1 * (np.linalg.norm(acc_exp[i]) - np.linalg.norm(acc_user[i]))**2) for i in range(len(acc_exp))])

    return {'ksi_total': weights['pose']*s_pose + weights['velocity']*s_velocity + weights['acceleration']*s_accel}