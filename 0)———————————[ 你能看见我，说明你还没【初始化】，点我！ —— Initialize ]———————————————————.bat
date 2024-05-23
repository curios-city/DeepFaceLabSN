@echo off
:: Use UTF-8 encoding without BOM
chcp 65001 > nul
setlocal
echo 修改注册表以显示/隐藏文件

REG ADD "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced" /v Hidden /t REG_DWORD /d 2 /f > nul

echo ok!

echo 更改记事本的字体设置

reg add "HKCU\Software\Microsoft\Notepad" /v "lfFaceName" /t REG_SZ /d "Consolas" /f > nul

echo ok!

echo 更改命令提示符的字体设置

reg add "HKCU\Console" /v "FaceName" /t REG_SZ /d "Consolas" /f > nul

echo ok!

echo 图形设置 --- 硬件加速gpu计划

reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v HwSchMode /t REG_DWORD /d 2 /f > nul

echo ok!

echo 稍后请重启计算机方可生效【好家伙！这东西还真的能提速】

echo.

echo 读取当前目录

set "currentDir=%CD%"
:: attrib +h "%~f0" 隐藏自身
echo ok!
echo 重启资源管理器
taskkill /f /im explorer.exe > nul
start explorer.exe
echo ok!
:: echo 等待确保资源管理器已启动
:: timeout /t 2 /nobreak > NUL
echo 正在收缩全部子菜单
attrib +h *.*
echo ok!
echo 正在显示需要的文件
attrib -h "0)———————————[ 展开 全部 —— Expand All ]————————————————————————————————.bat"
attrib -h "0)++++++++++++++++[ 最近 使用 —— Recent Tse ]+++++++++++++++++++++++++++++++++++++++++++++.bat"
attrib -h "0)———————————[ 显卡 设置 —— RG 开关 ]—————————————————————————————————.bat"
attrib -h "1)———————————[ 视频 处理 —— Video Tools ]———————————————————————————————.bat"
attrib -h "2)———————————[ SRC 处理 —— Data_src Tools ]——————————————————————————————.bat"
attrib -h "3)———————————[ DST 处理 —— Data_dst Tools ]——————————————————————————————.bat"
attrib -h "4)———————————[ 遮罩 处理 —— XSeg Tools ]———————————————————————————————.bat"
attrib -h "5)———————————[ 模型 训练 —— Train Models ]———————————————————————————————.bat"
attrib -h "6)———————————[ 模型 应用 —— Merge Tools ]———————————————————————————————.bat"
attrib -h "7)———————————[ 封装 视频 —— Encode Videos ]——————————————————————————————.bat"
attrib -h "8)———————————[ 其他 测试 —— Extra Function ]——————————————————————————————.bat"
echo ok!
echo 打开DeepFaceLab目录
start "" "%currentDir%"

echo.
echo ----------------------------------------------------------------------------------------------------------
echo 1. 引言
echo.
echo 本免责声明适用于DeepFaceLab开源项目（以下简称“本项目”）。本项目是一个开放源代码的软件，旨在提供面部替换和图像处理的技术。用户（以下简称“您”）在使用本项目时，应仔细阅读并理解本免责声明中的所有条款。
echo.
echo 2. 许可范围
echo.
echo 本项目基于GPL-3.0 开源许可证发布。该许可证授权您使用、复制、修改和分发本项目，但必须符合该许可证的所有条款和条件。
echo.
echo 3. 免责声明
echo.
echo 本项目是在“现状”和“可用”的基础上提供的，不提供任何形式的明示或暗示保证，包括但不限于对适销性、特定用途的适用性或非侵权性的保证。在任何情况下，本项目的作者或版权所有者均不对因使用本项目而产生的任何直接、间接、偶然、特殊、示例性或后果性损害承担责任。
echo.
echo 4. 使用限制
echo.
echo 您在使用本项目时，必须遵守适用的法律和法规。您承诺不会将本项目用于任何非法或未经授权的目的，包括但不限于侵犯他人的版权、隐私权或其他权利。
echo.
echo 5. 版权和所有权
echo.
echo 本项目的版权和知识产权属于原作者。本免责声明不意味着转让任何版权或其他知识产权给用户。
echo.
echo 6. 最终解释权
echo.
echo 对于本免责声明的解释及其修改权归本项目的维护者所有。如果本免责声明的中英文版本出现冲突，以英文版本为准。
echo.
echo 汉化版：https://github.com/curios-city/DeepFaceLab
echo 英文版：https://github.com/MachineEditor/DeepFaceLab
echo.
echo 请选择是否开启欢迎界面：

set /p choice=输入您的选择（y/n）: 
set source=_internal\DeepFaceLab\utils\logo2.py
set destination=_internal\DeepFaceLab\utils\logo.py

if "%choice%"=="n" (
    echo Inside if block
    if exist %destination% (
        del %destination%
    )
    copy %source% %destination% > nul
) else (
    goto end
)

:end
endlocal
pause