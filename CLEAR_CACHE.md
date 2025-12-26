# 清除浏览器缓存指南

## 问题说明
Service Worker 缓存了旧版本的 CSS 文件路径（bootstrap.min.css 和 hux-blog.min.css），导致 404 错误。

## 已修复内容
1. ✅ 更新了 Service Worker 版本号（v1 → v2）
2. ✅ 添加了自动清除旧缓存的逻辑
3. ✅ 移除了 KaTeX 的 integrity 校验
4. ✅ 将 cdn.jsdelivr.net 添加到白名单

## 立即清除缓存的方法

### 方法 1：开发者工具清除（推荐）
1. 按 `F12` 打开开发者工具
2. 切换到 **Application** 标签（或 **应用程序**）
3. 在左侧找到 **Service Workers**
4. 点击 **Unregister** 注销当前的 Service Worker
5. 在 **Storage** 下找到 **Cache Storage**
6. 右键每个缓存项，选择 **Delete** 删除
7. 刷新页面（`Ctrl+Shift+R` 或 `Cmd+Shift+R`）

### 方法 2：浏览器清除缓存
1. Chrome/Edge: `Ctrl+Shift+Delete` (Mac: `Cmd+Shift+Delete`)
2. 选择 **缓存的图片和文件**
3. 时间范围选择 **全部时间**
4. 点击 **清除数据**
5. 重新访问网站

### 方法 3：无痕模式测试
1. `Ctrl+Shift+N` (Mac: `Cmd+Shift+N`) 打开无痕窗口
2. 访问您的网站
3. 查看问题是否解决

## 验证修复
打开浏览器控制台（F12），检查：
- ✅ 不再有 `bootstrap.min.css` 404 错误
- ✅ 不再有 `hux-blog.min.css` 404 错误
- ✅ 不再有 KaTeX integrity 错误
- ✅ 不再有 JSON.parse 语法错误

## 自动生效
下次访问时，新的 Service Worker (v2) 会自动：
- 删除旧版本的缓存
- 使用正确的 CSS 文件路径
- 正常加载所有资源

## 注意事项
- 每次更新后，强制刷新页面（`Ctrl+Shift+R`）
- GitHub Pages 可能需要 1-2 分钟部署时间
- 移动设备也需要清除缓存

---
修复时间：2025年12月26日
