
# 导入工具包
import requests

# 设置服务请求的地址URL
url = "http://0.0.0.0:5001/v1/recognition/"
data = {"text1":"人生该如何起头", "text2":"改变要如何起手"}
res = requests.post(url, data=data)

# 打印返回结果
print("预测样本:", data["text1"], "|", data["text2"])
print("预测结果:", res.text)

