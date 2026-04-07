import requests
import pandas as pd

# API URL
api_url = "https://www.dongqiudi.com/sport-data/soccer/biz/data/standing?season_id=26350&app=dqd&version=0&platform=web&language=zh-cn&app_type="

# 伪装头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0",
    "Referer": "https://www.dongqiudi.com/data/24",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive"
}

print("🕵️‍♂️ 正在深入挖掘懂球帝底层的真实球队数据...")

try:
    response = requests.get(api_url, headers=headers, timeout=30)
    
    if response.status_code == 200:
        json_data = response.json()
        
        # 🎯 核心修正：按照“俄罗斯套娃”的结构，一层层深入到真正的球队列表
        # 第 0 个 round 通常代表最新的总积分榜
        true_team_data = json_data['content']['rounds'][0]['content']['data']
        
        # 将真正的球队列表转换成表格
        df = pd.DataFrame(true_team_data)
        
        print("\n🎉 破案了！终于抓到真正的球队数据啦！")
        
        # 挑选出最关键的列打印在终端让你预览
        # (team_name:队名, matches_total:场次, win:胜, draw:平, loss:负, goals_pro:进球, goals_against:失球, points:积分)
        core_columns = ['team_name', 'matches_total', 'win', 'draw', 'loss', 'goals_pro', 'goals_against', 'points']
        
        # 过滤出表格中实际存在的列进行打印防报错
        available_cols = [col for col in core_columns if col in df.columns]
        print("\n🏆 中甲最新战绩预览 (前五名):")
        print(df[available_cols].head())
        
        # 保存到你指定的绝对路径，直接覆盖刚才那个没用的文件
        save_path = "W:/Desktop/文件/001/zhongjia_real_data.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 真正的比赛数据已成功保存至：{save_path}")
        print("现在，你可以把这个完美的 CSV 文件喂给你的 PyTorch 预测模型了！")
        
    else:
        print(f"\n❌ 请求失败！状态码：{response.status_code}")

except Exception as e:
    print("\n⚠️ 提取数据遇到阻碍：", e)