#!/bin/zsh
# Transformer GPU Visual Simulator — 一键启动脚本
# 双击本文件即可启动本地服务器，然后在浏览器打开 http://localhost:5173/
# 关闭本终端窗口即可停止服务器。

cd "$(dirname "$0")"

echo "=============================================="
echo "  Transformer GPU Visual Simulator"
echo "=============================================="
echo "正在启动本地开发服务器..."
echo "启动成功后请在浏览器打开:  http://localhost:5173/"
echo "关闭本窗口即可停止服务器。"
echo "=============================================="
echo ""

npm run dev
