# step 1：导入 Neo4j 驱动包
from neo4j import GraphDatabase
# step 2：连接 Neo4j 图数据库
driver = GraphDatabase.driver("bolt://121.5.163.84:7687", auth=("neo4j", "bbmmze3e4bb!"),encrypted=False)

# 添加 关系 函数
def add_friend(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
        "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
        name=name, friend_name=friend_name)
# 定义 关系函数
def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                        "RETURN friend.name ORDER BY friend.name", name=name):
        print(record["friend.name"])

# step 3：运行
with driver.session() as session:
    session.write_transaction(add_friend, "Arthur", "Guinevere")
    session.write_transaction(add_friend, "Arthur", "Lancelot")
    session.write_transaction(add_friend, "Arthur", "Merlin")
    session.read_transaction(print_friends, "Arthur")