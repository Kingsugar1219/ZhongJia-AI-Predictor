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
        # 我们输入 6 个特征 (主队进球/失球/积分, 客队进球/失球/积分)
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

# 实例化 AI 大脑并设置为考试模式
model = FootballPredictorNN()
model.eval()

# ==========================================
# 2. 智能获取实时数据 (自动抓取 + 缓存双保险)
# ==========================================
# ttl=3600: 每一个小时才会真正在后台去抓取一次，其余时间秒开网页，绝对防封号！
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
        # 尝试去拉取最新数据
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            json_data = response.json()
            true_team_data = json_data['content']['rounds'][0]['content']['data']
            latest_df = pd.DataFrame(true_team_data)
            
            # 将最新数据保存一份到相对路径，作为断网时的保底备份
            latest_df.to_csv("zhongjia_real_data.csv", index=False, encoding='utf-8-sig')
            return latest_df
        else:
            # 接口拒绝访问时，使用备胎数据
            st.toast("⚠️ 实时接口暂不可用，正在使用最近一次的备份数据。")
            return pd.read_csv("zhongjia_real_data.csv")
            
    except Exception:
        # 网络超时或报错时，同样使用备胎数据防止网站崩溃
        st.toast("🔌 网络产生延迟，正在使用本地备份数据进行推演。")
        return pd.read_csv("zhongjia_real_data.csv")

# ==========================================
# 3. 网页主视觉与 UI 交互
# ==========================================
st.title("⚽ 中甲联赛 AI 胜率预测系统")
st.write("🧠 基于 PyTorch 深度学习框架 | 📡 全自动无感同步最新赛事数据")
st.divider()

try:
    # 直接调用函数，如果缓存没过期瞬间返回，过期了自动后台抓取
    df = load_latest_data()
    
    # 提取所有球队的名字供下拉菜单使用
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

        # 赛前数据看板
        st.subheader(f"📊 赛前状态对比")
        
        # 使用 metrics 展示数据更直观
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label=f"🏠 {home_team} 积分", value=home_stats['points'])
        m_col2.metric(label="⚔️ 进球对比 (主 VS 客)", value=f"{home_stats['goals_pro']} : {away_stats['goals_pro']}")
        m_col3.metric(label=f"✈️ {away_team} 积分", value=away_stats['points'])
        
        st.write("") # 留点空白
        
        # ==========================================
        # 4. 核心引擎：触发 PyTorch 计算
        # ==========================================
        # use_container_width=True 让按钮填满屏幕宽度，更好看
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
                
                # 展示高光预测结果
                st.divider()
                st.subheader("🏆 AI 终极预测结果")
                
                # 三个大卡片展示概率
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.success(f"🟢 **主胜概率**\n\n### {prob_home:.1f}%")
                with res_col2:
                    st.warning(f"🟡 **平局概率**\n\n### {prob_draw:.1f}%")
                with res_col3:
                    st.error(f"🔴 **客胜概率**\n\n### {prob_away:.1f}%")
                
                st.caption("注：当前预测基于 PyTorch 初始化模型状态。未来接入历史回放数据完成深度训练后，预测精度将大幅提升！")

except Exception as e:
    st.error(f"❌ 系统初始化遇到了一点小阻碍，可能缺少必要的底层数据文件。错误报告：{e}")
