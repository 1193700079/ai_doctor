# -*- coding:UTF-8 -*-
REDIS_CONFIG = {
     "host": "0.0.0.0",
     "port": 6379
}


NEO4J_CONFIG = {
    "uri": "bolt://121.43.138.58:7687",
    "auth": ("neo4j", "bbmmze3e4bb!"),
    "encrypted": False
}

model_serve_url = "http://121.43.138.58:5001/v1/recognition/"

TIMEOUT = 2

reply_path = "./reply.json"

# redis 过期时间
ex_time = 36000