import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 定义 PyTorch 神经网络模型
# ==========================================
class FootballPredictorNN(nn.Module):
    def __init__(self):
        super(FootballPredictorNN, self).__init__()
        # 我们输入 6 个特征 (主队进球/失球/积分, 客队进球/失球/积分)
        self.layer1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        # 输出 3 个结果类别：0(客队胜), 1(平局), 2(主队胜)
        self.output_layer = nn.Linear(8, 3) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

# 实例化 AI 大脑并设置为考试模式
model = FootballPredictorNN()
model.eval()

# ==========================================
# 2. 读取真实数据 (你刚刚抓取的 CSV)
# ==========================================
# 使用 st.cache_data 让 Streamlit 记住这份数据，不用每次点按钮都去读硬盘
@st.cache_data 
def load_data():
    # 注意：确保这个路径和你的 csv 文件路径完全一致！
    return pd.read_csv("W:/Desktop/文件/001/zhongjia_real_data.csv")

st.title("⚽ 中甲联赛 AI 胜率预测系统")

import requests

# 在网页上放一个刷新按钮
if st.button("🔄 一键抓取懂球帝最新数据"):
    with st.spinner("🕵️‍♂️ 正在潜入后台拉取最新中甲积分榜..."):
        api_url = "https://www.dongqiudi.com/sport-data/soccer/biz/data/standing?season_id=26350&app=dqd&version=0&platform=web&language=zh-cn&app_type="
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Referer": "https://www.dongqiudi.com/data/24"
        }
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                json_data = response.json()
                true_team_data = json_data['content']['rounds'][0]['content']['data']
                new_df = pd.DataFrame(true_team_data)
                
                # 直接覆盖保存到本地
                new_df.to_csv("W:/Desktop/文件/001/zhongjia_real_data.csv", index=False, encoding='utf-8-sig')
                
                # 清除 Streamlit 的缓存，强制重新读取新数据
                st.cache_data.clear() 
                st.success("✅ 数据更新成功！现在使用的是最新的赛后数据！")
        except Exception as e:
            st.error(f"❌ 更新失败：{e}")
st.write("基于 PyTorch 深度学习与最新实时赛事数据")
st.divider()

try:
    df = load_data()
    # 提取所有球队的名字供下拉菜单使用
    team_names = df['team_name'].tolist()
    
    # ==========================================
    # 3. 网页 UI：选择对抗球队
    # ==========================================
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 选择主场球队", team_names, index=0)
    with col2:
        away_team = st.selectbox("✈️ 选择客场球队", team_names, index=1)

    if home_team == away_team:
        st.warning("⚠️ 主队和客队不能是同一支球队哦！请重新选择。")
    else:
        # 从表格中抽取出被选中的两支球队的具体数据
        home_stats = df[df['team_name'] == home_team].iloc[0]
        away_stats = df[df['team_name'] == away_team].iloc[0]

        # 在网页上展示当前的数据对比
        st.subheader(f"📊 赛前数据对比：{home_team} VS {away_team}")
        st.write(f"**{home_team} (主)**: 当前积分 {home_stats['points']} | 进球 {home_stats['goals_pro']} | 失球 {home_stats['goals_against']}")
        st.write(f"**{away_team} (客)**: 当前积分 {away_stats['points']} | 进球 {away_stats['goals_pro']} | 失球 {away_stats['goals_against']}")
        
        # ==========================================
        # 4. 核心：调用 PyTorch 进行预测
        # ==========================================
        if st.button("🔮 让 AI 大脑预测胜率", type="primary"):
            with st.spinner("🧠 AI 正在进行战术推演..."):
                
                # 组装特征：[主进球, 主失球, 主积分, 客进球, 客失球, 客积分]
                features = [
                    float(home_stats['goals_pro']), float(home_stats['goals_against']), float(home_stats['points']),
                    float(away_stats['goals_pro']), float(away_stats['goals_against']), float(away_stats['points'])
                ]
                
                # 转换成 PyTorch 需要的张量 (Tensor)
                input_tensor = torch.tensor([features], dtype=torch.float32)
                
                # 开始预测
                with torch.no_grad():
                    output = model(input_tensor)
                    # 使用 Softmax 函数，把输出的绝对数值转换成加起来等于 100% 的概率
                    probabilities = F.softmax(output, dim=1).numpy()[0]
                
                # 提取各项概率
                prob_away = probabilities[0] * 100
                prob_draw = probabilities[1] * 100
                prob_home = probabilities[2] * 100
                
                st.divider()
                st.subheader("🏆 预测结果揭晓")
                st.info(f"🟢 **{home_team} 胜率**: {prob_home:.2f}%")
                st.warning(f"🟡 **平局概率**: {prob_draw:.2f}%")
                st.error(f"🔴 **{away_team} 胜率**: {prob_away:.2f}%")
                
                st.caption("注：由于目前尚未导入历史赛程进行权重训练，当前胜率为 PyTorch 模型的初始推演展示。")

except FileNotFoundError:
    st.error("❌ 找不到 CSV 文件！请确认 'zhongjia_real_data.csv' 和这段代码在一个文件夹里。")