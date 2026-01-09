import pandas as pd
import numpy as np
from collections import Counter

# 데이터 로드
df = pd.read_csv('open_track1/train.csv')

print("=" * 80)
print("데이터 기본 통계")
print("=" * 80)
print(f"총 경기 수: {df['game_id'].nunique():,}")
print(f"총 에피소드 수: {df['game_episode'].nunique():,}")
print(f"총 이벤트 수: {len(df):,}")
print(f"경기당 평균 이벤트 수: {len(df) / df['game_id'].nunique():.1f}")
print(f"에피소드당 평균 이벤트 수: {len(df) / df['game_episode'].nunique():.1f}")

print("\n" + "=" * 80)
print("이벤트 타입 분포")
print("=" * 80)
type_counts = df['type_name'].value_counts()
print(type_counts)
print(f"\n이벤트 타입 수: {df['type_name'].nunique()}")

print("\n" + "=" * 80)
print("결과(Result) 분포")
print("=" * 80)
result_counts = df['result_name'].value_counts()
print(result_counts)
print(f"\n결과 타입 수: {df['result_name'].nunique()}")

# 존(Zone) 계산 - 피치를 그리드로 나눔
def calculate_zone(x, y, x_bins=5, y_bins=4):
    """좌표를 Zone으로 변환 (예시: 5x4 그리드)"""
    x_zone = min(int(x / (105 / x_bins)), x_bins - 1)
    y_zone = min(int(y / (68 / y_bins)), y_bins - 1)
    return f"Zone_{x_zone}_{y_zone}"

df['start_zone'] = df.apply(lambda row: calculate_zone(row['start_x'], row['start_y']), axis=1)
zone_counts = df['start_zone'].value_counts()
print("\n" + "=" * 80)
print(f"존(Zone) 분포 (5x4 그리드 = 20개 존)")
print("=" * 80)
print(zone_counts)

# 토큰 조합 분석
print("\n" + "=" * 80)
print("토큰화 시뮬레이션 (타입 + 결과 + 존)")
print("=" * 80)

# NaN 처리
df_with_result = df[df['result_name'].notna()].copy()
df_with_result['token'] = (
    df_with_result['type_name'].astype(str) + "_" +
    df_with_result['result_name'].astype(str) + "_" +
    df_with_result['start_zone'].astype(str)
)

token_counts = df_with_result['token'].value_counts()
print(f"총 고유 토큰 수: {len(token_counts)}")
print(f"이론적 최대 토큰 수: {df['type_name'].nunique()} × {df_with_result['result_name'].nunique()} × {df['start_zone'].nunique()} = {df['type_name'].nunique() * df_with_result['result_name'].nunique() * df['start_zone'].nunique()}")

# 희소성 분석
print(f"\n토큰 등장 빈도 통계:")
print(f"  평균 등장 횟수: {token_counts.mean():.1f}")
print(f"  중앙값 등장 횟수: {token_counts.median():.1f}")
print(f"  최소 등장 횟수: {token_counts.min()}")
print(f"  최대 등장 횟수: {token_counts.max():,}")

# 등장 빈도별 토큰 분포
freq_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
freq_labels = ['1-9회', '10-49회', '50-99회', '100-499회', '500-999회', '1000+회']
token_freq_dist = pd.cut(token_counts, bins=freq_bins, labels=freq_labels).value_counts().sort_index()

print(f"\n토큰 희소성 분석:")
for label, count in token_freq_dist.items():
    percentage = (count / len(token_counts)) * 100
    print(f"  {label:12s}: {count:4d}개 ({percentage:5.1f}%)")

# 희소 토큰 비율
sparse_tokens = token_counts[token_counts < 50]
print(f"\n50회 미만 등장 토큰: {len(sparse_tokens)}개 ({len(sparse_tokens)/len(token_counts)*100:.1f}%)")
print(f"10회 미만 등장 토큰: {len(token_counts[token_counts < 10])}개 ({len(token_counts[token_counts < 10])/len(token_counts)*100:.1f}%)")

print("\n" + "=" * 80)
print("대안 1: 타입 + 결과만 결합 (존은 별도)")
print("=" * 80)
df_with_result['token_simple'] = (
    df_with_result['type_name'].astype(str) + "_" +
    df_with_result['result_name'].astype(str)
)
token_simple_counts = df_with_result['token_simple'].value_counts()
print(f"총 고유 토큰 수: {len(token_simple_counts)}")
print(f"평균 등장 횟수: {token_simple_counts.mean():.1f}")
print(f"중앙값 등장 횟수: {token_simple_counts.median():.1f}")
print(f"\n상위 10개 토큰:")
print(token_simple_counts.head(10))

sparse_simple = token_simple_counts[token_simple_counts < 50]
print(f"\n50회 미만 등장 토큰: {len(sparse_simple)}개 ({len(sparse_simple)/len(token_simple_counts)*100:.1f}%)")

print("\n" + "=" * 80)
print("결론 및 추천")
print("=" * 80)
