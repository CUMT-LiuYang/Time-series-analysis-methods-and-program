




import pandas as pd

# 读取CSV文件

df = pd.read_csv("../../2022标准化数据.csv")

# 合并日期和时间列
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')

# 将datetime列格式化为"%Y-%m-%dT%H:%M:%SZ"
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# 删除原始的日期和时间列
df = df.drop(columns=['date', 'hour'])

# 重新排列列的顺序，将datetime列放在第一列
try:
    df = df[['datetime', 'CO', 'NO2', 'O3', 'SO2']]
    print("Unstandard Format")
except:
    df = df[['datetime', 'CO (mg/m³)', 'NO2 (µg/m³)', 'O3 (µg/m³)', 'SO2 (µg/m³)']]
    print("Standard Format")
# 输出到CSV文件
df.to_csv("time_integration.csv", index=False)

print("结果已保存到time_add.csv文件中。")
