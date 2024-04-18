import pandas as pd
from sklearn.linear_model import LinearRegression

# 创建示例数据
data = {
    '情绪指数':[0.08,0.11, 0.07,],
    '上证指数收盘价' : [3255.67, 3269.32,3264.81,]
}
df = pd.DataFrame(data)

#将数据分为特征变量X和目标变量y
X = df[['情绪指数']]
y = df['上证指数收盘价']

# 建立线性回归模型
model = LinearRegression()
model.fit(X,y)

# 计算R^2
r_squared = model.score(X,y)
print(f"R^2(决定系数):{r_squared:.4f}")

# 输出模型的系数和截距
print(f"系数:{model.coef_[0]:.2f}")
print(f"截距:{model.intercept_:.2f}")

#使用模型进行预测
sentiment_score =0.75
predicted_close_price = model.predict([[sentiment_score]])
print(f"当情绪指数为{sentiment_score}时,预测的上证指数收盘价为:{predicted_close_price[0]:.2f}")

