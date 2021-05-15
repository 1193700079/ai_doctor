# coding: utf-8
# Team : Quality Management Center
# Author：RuiqingY
# Date ：2021/4/16 17:09
# Tool ：PyCharm
# -*- coding=utf-8 -*-
# 引入相关包
import os
import fileinput
from neo4j import GraphDatabase
# from config import NEO4J_CONFIG
# NEO4J_CONFIG = {
#     "uri": "bolt://121.43.138.58:7687",
#     "auth": ("neo4j", "bbmmze3e4bb!"),
#     "encrypted": False
# }
# driver = GraphDatabase.driver( **NEO4J_CONFIG)
driver = GraphDatabase.driver("bolt://121.5.163.84:7687", auth=("neo4j", "bbmmze3e4bb!"),encrypted=False)


def query_neo4j(text):
    """
    description: 根据用户对话文本中的可能存在的症状查询图数据库.
    :param text: 用户的输入文本.
    :return: 用户描述的症状对应的疾病列表.
    """
    # 开启一个session操作图数据库
    with driver.session() as session:
         # cypher语句, 匹配句子中存在的所有症状节点,
         # 保存这些节点并逐一通过关系dis_to_sym进行对应病症的查找, 返回找到的疾病名字列表.
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH \
                  a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" %text
        # 运行这条cypher语句
        record = session.run(cypher)
        print("运行查询cypher语句完毕", record)
        # 从record对象中获得结果列表
        result = list(map(lambda x: x[0], record))
        print("结果列表", result)
    return result
if __name__ == '__main__':

    query_neo4j("耳鸣")