import time
import yfinance as yf
import pandas as pd
from datetime import datetime

def get_stock_info(ticker_symbol):
    """
    """
    try:
        # 创建股票ticker对象
        stock = yf.Ticker(ticker_symbol)

        # 获取最近两个交易日的数据
        historical_data = stock.history(period="2d")

        # 如果没有获取到数据
        if historical_data.empty:
            return {"error1": f"无法获取 {ticker_symbol} 的股价数据"}

        # 获取最新交易日数据
        latest_data = historical_data.iloc[-1]
        latest_date = historical_data.index[-1].strftime('%Y-%m-%d')

        # 准备结果字典
        result = {
            "代码": ticker_symbol,
            "日期": latest_date,
            "价格": round(latest_data['Close'], 2),
            "开盘价": round(latest_data['Open'], 2),
            "最高价": round(latest_data['High'], 2),
            "最低价": round(latest_data['Low'], 2),
            "交易量": int(latest_data['Volume'])
        }

        # 如果有足够的历史数据计算涨跌幅
        if len(historical_data) >= 2:
            previous_close = historical_data['Close'].iloc[-2]
            price_change = latest_data['Close'] - previous_close
            price_change_percent = (price_change / previous_close) * 100

            result.update({
                "前收盘价": round(previous_close, 2),
                "额": round(price_change, 2),
                "幅": round(price_change_percent, 2)
            })
        else:
            result.update({
                "额": 0,
                "幅": 0
            })

        # 尝试获取公司名称
        try:
            company_info = stock.info
            if 'shortName' in company_info:
                result["名称"] = company_info['shortName']
        except:
            result["名称"] = "未知"

        return result

    except Exception as e:
        return {"error2": f"获取 {ticker_symbol} 信息时出错: {str(e)}"}

# 使用示例
def print_stock_info(ticker_symbol):
    """打印股票信息的格式化输出"""
    info = get_stock_info(ticker_symbol)

    if "error" in info:
        print("error3")
        print(info["error"])
        return

    # 判断股票市场，设置货币单位
    currency = "G" if ".HK" in ticker_symbol else "D" if "." not in ticker_symbol else "R"

    code1 = f"{info['代码']}"
    code2 = f"{info['名称']}"
    code3 = f"{info['日期']}"
    code4 = f"{info['价格']} {currency}"
    code5 = f"{info['幅']}%"
    code6 = "📈" if info["额"] > 0 else "📉" if info["额"] < 0 else "⬜"

    print(code1 + "  " + code2+ "  "  + code3+ "  "  + code4+ "  "  + code6 + code5)

def run_test(string_list):
    for cd in string_list:
        if "-" == cd:
            print("----------------")
        else:
            print_stock_info(cd)





def run_table(string_list):
    data = {
        "aa": [],
        "bb": [],
        "cc": [],
        "dd": [],
        "ee": []
    }

    for cd in string_list:
        if "-" in cd:
             data["aa"].append(cd)
             data["bb"].append("")
             data["cc"].append("")
             data["dd"].append("")
             data["ee"].append("")
             continue

        info = get_stock_info(cd)
        data["aa"].append(info["代码"])
        currency = "G" if ".HK" in info["代码"] else "D" if "." not in info["代码"] else "R"

        data["bb"].append(info["名称"])
        data["cc"].append(info["日期"])
        data["dd"].append(str(info["价格"]) + currency)

        code6 = "📈" if info["额"] > 0 else "📉" if info["额"] < 0 else "⬜"
        data["ee"].append(str(info["幅"]) + code6)
    df = pd.DataFrame(data)
    #display(df)

    print(df.to_string(header=False))


# 使用示例
if __name__ == "__main__":
    string_list = ["0700.HK","603605.SS","000333.SZ","000066.SZ","3690.HK", "300750.SZ","605117.SS",
               "-SM--","9988.HK","1211.HK","600089.SS","1810.HK","601288.SS","600809.SS","002050.SZ",
               "-BK--","601988.SS","601939.SS",
               "-YOUYUN--","601857.SS","600941.SS","603288.SS","000651.SZ","688047.SS","0728.HK",
               "-US--","TSLA","NVDA","AMD",]
    print(datetime.now())
    run_table(string_list)

    data = yf.Ticker("^GSPC").history(period="1mo")
    print(data.head())
