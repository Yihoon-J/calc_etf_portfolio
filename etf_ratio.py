import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

rat_low=0 #개별 자산의 비중 하한
rat_hi=1 #개별 자산의 비중 상한

def analyze_portfolio(file_path):
    """
    포트폴리오 분석 및 최적화를 수행하는 함수
    
    Parameters:
    file_path (str): CSV 파일 경로
    
    Returns:
    tuple: 최적 포트폴리오 비중, 수익률, 리스크, 상관계수 행렬
    """
    
    # 1. 데이터 로드
    df = pd.read_csv(file_path, encoding='cp949', thousands=",")
    
    # 2. 수익률 계산
    price_cols = ['TIGER S&P','ACE Gold', 'TIGER NIFTY']
    returns = df[price_cols].pct_change().dropna()
    
    # 3. 연간화된 통계량 계산 (252 거래일 기준)
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # 4. 상관계수 행렬 계산
    correlation = returns.corr()
    
    # 5. 포트폴리오 최적화 함수들
    def portfolio_metrics(weights):
        """포트폴리오의 수익률과 변동성을 계산"""
        portfolio_return = np.sum(annual_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_std
    
    def min_variance_objective(weights):
        """최소분산 포트폴리오를 위한 목적함수"""
        return portfolio_metrics(weights)[1]
    
    def sharpe_ratio_objective(weights):
        """샤프 비율 최대화를 위한 목적함수 (무위험수익률 = 3% 사용)"""
        rf = 0.03  # 무위험수익률 3%
        portfolio_return, portfolio_std = portfolio_metrics(weights)
        return -(portfolio_return - rf) / portfolio_std
    
    # 6. 최적화 제약조건 설정
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 비중의 합 = 1
    ]
    
    # 각 자산의 비중 제한 X
    bounds = tuple((rat_low, rat_hi) for _ in range(len(price_cols)))
    
    # 7. 최소분산 포트폴리오 최적화
    min_var_result = minimize(min_variance_objective,
                            [1./len(price_cols)] * len(price_cols),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
    
    min_var_weights = min_var_result.x
    min_var_return, min_var_std = portfolio_metrics(min_var_weights)
    
    # 8. 샤프비율 최대화 포트폴리오 최적화
    sharpe_result = minimize(sharpe_ratio_objective,
                           [1./len(price_cols)] * len(price_cols),
                           method='SLSQP',
                           bounds=bounds,
                           constraints=constraints)
    
    sharpe_weights = sharpe_result.x
    sharpe_return, sharpe_std = portfolio_metrics(sharpe_weights)
    
    # 9. 결과 출력
    print("\n=== 개별 자산 통계 ===")
    for asset in price_cols:
        print(f"\n{asset}:")
        print(f"연간 수익률: {annual_returns[asset]:.2%}")
        print(f"연간 변동성: {np.sqrt(cov_matrix.loc[asset, asset]):.2%}")
    
    print("\n=== 상관계수 ===")
    print(correlation.round(3))
    
    print("\n=== 최소분산 포트폴리오 ===")
    for asset, weight in zip(price_cols, min_var_weights):
        print(f"{asset}: {weight:.1%}")
    print(f"예상 연간 수익률: {min_var_return:.2%}")
    print(f"예상 연간 변동성: {min_var_std:.2%}")
    
    print("\n=== 최대 샤프비율 포트폴리오 ===")
    for asset, weight in zip(price_cols, sharpe_weights):
        print(f"{asset}: {weight:.1%}")
    print(f"예상 연간 수익률: {sharpe_return:.2%}")
    print(f"예상 연간 변동성: {sharpe_std:.2%}")
    
    # 10. 시각화
    # 상관계수 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlations')
    plt.show()
    
    # 누적수익률 라인차트
    cumulative_returns = (1 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot()
    plt.title('Cumm.Returns')
    plt.grid(True)
    plt.show()
    
    # 11. 3D 효율적 프론티어 시각화
    import plotly.graph_objects as go
    import itertools

    def generate_portfolio_surface(returns, cov_matrix, num_points=20):
        
        # 그리드 생성
        w1_range = np.linspace(rat_low, rat_hi, num_points)
        w2_range = np.linspace(rat_low, rat_hi, num_points)
        W1, W2 = np.meshgrid(w1_range, w2_range)
        
        # 변동성 저장할 배열
        V = np.zeros_like(W1)
        W3 = 1 - W1 - W2  # 나머지 비중
        
        # 유효한 포트폴리오 영역 계산
        valid_portfolios = (W3 >= rat_low) & (W3 <= rat_hi)
        
        # 각 격자점에서 포트폴리오 변동성 계산
        for i in range(num_points):
            for j in range(num_points):
                if valid_portfolios[i,j]:
                    weights = np.array([W1[i,j], W2[i,j], W3[i,j]])
                    V[i,j] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                else:
                    V[i,j] = np.nan  # 유효하지 않은 영역
        
        return W1, W2, W3, V

    # 포트폴리오 조합 생성
    w1_grid, w2_grid, w3_grid, volatility_grid = generate_portfolio_surface(annual_returns, cov_matrix)

    # Surface 플롯 생성
    fig = go.Figure()
    
    # Surface 추가
    fig.add_trace(go.Surface(
        x=w1_grid,
        y=w3_grid,
        z=volatility_grid,
        colorscale='Viridis',
        colorbar=dict(title='포트폴리오 변동성'),
        name='Portfolio Surface'
    ))

    # 최소분산 포트폴리오
    fig.add_trace(go.Scatter3d(
        x=[min_var_weights[0]],
        y=[min_var_weights[2]],
        z=[min_var_result.fun],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond'
        ),
        name='최소 분산 포트폴리오'
    ))

    # 최대 샤프비율 포트폴리오
    fig.add_trace(go.Scatter3d(
        x=[sharpe_weights[0]],
        y=[sharpe_weights[2]],
        z=[portfolio_metrics(sharpe_weights)[1]],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='diamond'
        ),
        name='최대 초과수익률 포트폴리오'
    ))

    fig.update_layout(
        scene = dict(
            xaxis_title='S&P 비중',
            yaxis_title='금현물 비중',
            zaxis_title='포트폴리오 변동성',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        title='Portfolio Volatility Surface',
        showlegend=True
    )
    
    fig.show()
    
    return min_var_weights, sharpe_weights, correlation, fig

if __name__ == "__main__":
    file_path = "etf_prices.csv" 
    min_var_weights, sharpe_weights, correlation, fig= analyze_portfolio(file_path)