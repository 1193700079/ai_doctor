# -*- coding:UTF-8 -*-
# 导入werobot和发送请求的requests
import werobot
import requests
# 主要逻辑服务请求地址
url = "http://localhost:5000/v1/main_serve/"
# 服务超时时间
TIMEOUT = 3
# 声明微信访问请求【框架将辅助完成微信联通验证】
robot = werobot.WeRoBot(token="itcast")
# 设置所有请求（包含文本、语音、图片等消息）入口
@robot.handler
def doctor(message, session):
    try:
        # 获得用户uid
        uid = message.source
        try:
            # 检查session，判断该用户是否第一次发言
            # 初始session为{}
            # 如果session中没有{uid:"1"}
            if session.get(uid, None) != "1":
                # 将添加{uid:"1"}
                session[uid] = "1"
                # 并返回打招呼用语
                return '您好, 我是智能医生小卿, 有什么需要帮忙的吗?'
            # 获取message中的用户发言内容
            text = message.content
        except:
            # 这里使用try...except是因为我用户很可能出现取消关注又重新关注的现象
            # 此时通过session判断，该用户并不是第一次发言，会获取message.content
            # 但用户其实又没有说话, 获取message.content时会报错
            # 该情况也是直接返回打招呼用语
            return '您好, 我是智能医生小卿, 有什么需要帮忙的吗 ?'
        # 获得发送主要逻辑服务的数据体
        data = {"uid": uid, "text": text}
        # 向主要逻辑服务发送post请求
        res = requests.post(url, data=data, timeout=TIMEOUT)
        # 返回主要逻辑服务的结果
        return res.text
    except Exception as e:
        print("出现异常:", e)
        return "对不起, 机器人客服正在休息..."
# 让服务器监听在 0.0.0.0:80
robot.config["HOST"] = "0.0.0.0"
robot.config["PORT"] = 80
robot.run()