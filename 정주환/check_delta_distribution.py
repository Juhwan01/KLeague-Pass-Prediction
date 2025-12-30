import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
train_df = pd.read_csv("open_track1/train.csv")

# Delta 계산
deltas_x = []
deltas_y = []

for name, group in train_df.groupby('game_episode'):
    if len(group) < 2:
        continue

    last = group.iloc[-1]
    penultimate = group.iloc[-2]

    delta_x = (last['end_x'] - penultimate['end_x']) / 105.0
    delta_y = (last['end_y'] - penultimate['end_y']) / 68.0

    deltas_x.append(delta_x)
    deltas_y.append(delta_y)

deltas_x = np.array(deltas_x)
deltas_y = np.array(deltas_y)

print("="*70)
print("Delta 분포 분석")
print("="*70)
print(f"\nDelta X:")
print(f"  Mean: {deltas_x.mean():.4f}")
print(f"  Std: {deltas_x.std():.4f}")
print(f"  Min: {deltas_x.min():.4f}")
print(f"  Max: {deltas_x.max():.4f}")
print(f"  -1~1 범위 내: {((deltas_x >= -1) & (deltas_x <= 1)).mean()*100:.2f}%")

print(f"\nDelta Y:")
print(f"  Mean: {deltas_y.mean():.4f}")
print(f"  Std: {deltas_y.std():.4f}")
print(f"  Min: {deltas_y.min():.4f}")
print(f"  Max: {deltas_y.max():.4f}")
print(f"  -1~1 범위 내: {((deltas_y >= -1) & (deltas_y <= 1)).mean()*100:.2f}%")

print(f"\n⚠️ 범위 초과:")
print(f"  Delta X > 1: {(deltas_x > 1).sum()} ({(deltas_x > 1).mean()*100:.2f}%)")
print(f"  Delta X < -1: {(deltas_x < -1).sum()} ({(deltas_x < -1).mean()*100:.2f}%)")
print(f"  Delta Y > 1: {(deltas_y > 1).sum()} ({(deltas_y > 1).mean()*100:.2f}%)")
print(f"  Delta Y < -1: {(deltas_y < -1).sum()} ({(deltas_y < -1).mean()*100:.2f}%)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(deltas_x, bins=100, edgecolor='black', alpha=0.7)
axes[0].axvline(-1, color='red', linestyle='--', label='-1 (Tanh 하한)')
axes[0].axvline(1, color='red', linestyle='--', label='1 (Tanh 상한)')
axes[0].set_xlabel('Delta X')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Delta X Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(deltas_y, bins=100, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(-1, color='red', linestyle='--', label='-1 (Tanh 하한)')
axes[1].axvline(1, color='red', linestyle='--', label='1 (Tanh 상한)')
axes[1].set_xlabel('Delta Y')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Delta Y Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delta_distribution_check.png', dpi=150)
plt.show()

print("\n✅ 분석 완료: delta_distribution_check.png 저장")
