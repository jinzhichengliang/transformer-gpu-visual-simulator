import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  // 构建产物部署到 GitHub Pages 子路径；本地开发仍从根路径访问
  base: command === 'build' ? '/transformer-gpu-visual-simulator/' : '/',
  plugins: [react()],
  server: {
    // 绑定所有接口（IPv4 + IPv6），避免只绑定 IPv6 回环地址导致
    // 部分浏览器通过 localhost（127.0.0.1）访问失败的问题
    host: true,
    port: 5173,
    strictPort: true,
  },
}))
