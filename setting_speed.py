import os
ini = "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple/\n"
pippath=os.environ["USERPROFILE"]+"\\pip\\"
exec("if not os.path.exists(pippath):\n\tos.mkdir(pippath)")
open(pippath+"/pip.ini","w+").write(ini)