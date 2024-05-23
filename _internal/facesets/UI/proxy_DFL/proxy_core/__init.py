# -*- coding: utf-8 -*-
import os
import sys
import configparser

"""
# 运行时路径。并非__init__.py的路径
BASE_DIR = "..\\business_platform"
t=os.path.dirname(os.path.dirname(os.path.realpath(BASE_DIR)))
if Path(BASE_DIR).exists():
    sys.path.append(BASE_DIR)
else:
    # 尝试下探一级路径
    sys.path.append("..\\..\\business_platform")
"""

#from UI.proxy_DFL import proxy_models as models

DFLconfigPath = ".\\UI\\proxy_DFL\\config.ini"
DFLconf = configparser.ConfigParser()
DFLpath=None
try:
    """
    if not os.path.exists(DFLconfigPath) or not os.path.isfile(DFLconfigPath):
                        open(DFLconfigPath, "w").close()
                        DFLconf.add_section("globle")
                        DFLconf.set("globle","DFLpath",".\\")
                        DFLconf.write(open(DFLconfigPath, "w"))
    """
    DFLconf.read(DFLconfigPath)
    DFLlocation=DFLconf.get("globle","DFLpath")
    DFLlocationRealtime=os.path.realpath(DFLlocation)
    DFLpath=os.path.join(DFLlocationRealtime,"_internal","DeepFaceLab")

except Exception as e:
    print(e.__str__())
try:
    sys.path.index(DFLpath)
except Exception:
    sys.path.append(DFLpath)
    str1=sys.path.__str__()
    print(str1)

from core.leras import nn
from core import pathex
from core import imagelib
from core.interact import interact