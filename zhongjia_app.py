import os
import streamlit as st
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 网页基础配置 (一定要写在最前面)
# ==========================================
st.set_page_config(page_title="中甲 AI 预测系统", page_icon="⚽", layout="centered")

# ==========================================
# 1. 定义 PyTorch 神经网络模型
# ==========================================
class FootballPredictorNN(nn.Module):
    def __init__(self):
        super(FootballPredictorNN, self).__init__()
        # 输入 6 个特征 (主进球/失球/积分, 客进球/失球/积分)
        self.layer1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        # 输出 3 个结果类别：0(客负), 1(平局), 2(主胜)
        self.output_layer = nn.Linear(8, 3) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

# ==========================================
# 2. 工业级模型加载方案 (带缓存与 UI 状态指示灯)
# ==========================================
@st.cache_resource
def load_ai_brain():
    brain = FootballPredictorNN()
    # 检查云端目录下到底有没有这个权重文件
    if os.path.exists("zhongjia_brain.pth"):
        try:
            # 尝试注入训练好的权重
            brain.load_state_dict(torch.load("zhongjia_brain.pth", weights_only=False))
            status_msg = "✅ [系统状态] 历史智慧大脑已成功连接！"
        except Exception as e:
            status_msg = f"❌ [系统状态] 大脑文件损坏或版本不符：{e}"
    else:
        status_msg = "⚠️ [系统状态] 未检测到 zhongjia_brain.pth 文件，当前为随机瞎猜状态！"
    
    brain.eval()
    return brain, status_msg

# ==========================================
# 3. 智能获取实时数据 (自动抓取 + 缓存双保险)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="🕵️‍♂️ 正在潜入后台自动同步中甲最新实时数据...") 
def load_latest_data():
    api_url = "https://www.dongqiudi.com/sport-data/soccer/biz/data/standing?season_id=26350&app=dqd&version=0&platform=web&language=zh-cn&app_type="
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "Referer": "https://www.dongqiudi.com/data/24",
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive"
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            json_data = response.json()
            true_team_data = json_data['content']['rounds'][0]['content']['data']
            latest_df = pd.DataFrame(true_team_data)
            latest_df.to_csv("zhongjia_real_data.csv", index=False, encoding='utf-8-sig')
            return latest_df
        else:
            st.toast("⚠️ 实时接口暂不可用，正在使用最近一次的备份数据。")
            return pd.read_csv("zhongjia_real_data.csv")
    except Exception:
        st.toast("🔌 网络产生延迟，正在使用本地备份数据进行推演。")
        return pd.read_csv("zhongjia_real_data.csv")

# ==========================================
# 4. 网页主视觉与 UI 交互
# ==========================================
st.title("⚽ 中甲联赛 AI 胜率预测系统")
st.write("🧠 基于 PyTorch 深度学习框架 | 📡 全自动无感同步最新赛事数据")

# 加载模型并打印状态灯（直接显示在标题下方）
model, load_status = load_ai_brain()
if "✅" in load_status:
    st.success(load_status)
elif "❌" in load_status:
    st.error(load_status)
else:
    st.warning(load_status)

st.divider()

try:
    # 获取数据
    df = load_latest_data()
    team_names = df['team_name'].tolist()
    
    # 选择对抗球队
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 选择主场球队", team_names, index=0)
    with col2:
        away_team = st.selectbox("✈️ 选择客场球队", team_names, index=1)

    if home_team == away_team:
        st.warning("⚠️ 左右互搏是不行的哦！请为客场重新选择一支球队。")
    else:
        # 提取两支球队的统计数字
        home_stats = df[df['team_name'] == home_team].iloc[0]
        away_stats = df[df['team_name'] == away_team].iloc[0]

        # 赛前数据看板 (恢复为你要求的纯文本排版)
        st.subheader(f"📊 赛前数据对比：{home_team} VS {away_team}")
        st.write(f"**{home_team} (主)**: 当前积分 {home_stats['points']} | 进球 {home_stats['goals_pro']} | 失球 {home_stats['goals_against']}")
        st.write(f"**{away_team} (客)**: 当前积分 {away_stats['points']} | 进球 {away_stats['goals_pro']} | 失球 {away_stats['goals_against']}")
        
        st.write("") # 留白
        
        # ==========================================
        # 5. 核心引擎：触发 PyTorch 计算
        # ==========================================
        if st.button("🔮 启动 AI 战术推演引擎", type="primary", use_container_width=True):
            with st.spinner("🧠 神经网络正在计算特征权重..."):
                
                # 组装特征输入张量
                features = [
                    float(home_stats['goals_pro']), float(home_stats['goals_against']), float(home_stats['points']),
                    float(away_stats['goals_pro']), float(away_stats['goals_against']), float(away_stats['points'])
                ]
                input_tensor = torch.tensor([features], dtype=torch.float32)
                
                # 开始推理
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1).numpy()[0]
                
                prob_away = probabilities[0] * 100
                prob_draw = probabilities[1] * 100
                prob_home = probabilities[2] * 100
                
                # 展示预测结果
                st.divider()
                st.subheader("🏆 AI 终极预测结果")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.success(f"🟢 **主胜概率**\n\n### {prob_home:.1f}%")
                with res_col2:
                    st.warning(f"🟡 **平局概率**\n\n### {prob_draw:.1f}%")
                with res_col3:
                    st.error(f"🔴 **客胜概率**\n\n### {prob_away:.1f}%")

except Exception as e:
    st.error(f"❌ 系统初始化遇到了一点小阻碍，可能缺少必要的底层数据文件。错误报告：{e}")
