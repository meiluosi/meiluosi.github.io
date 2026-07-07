import type { FooterConfig } from "../types/footerConfig";

export const footerConfig: FooterConfig = {
	// 是否启用Footer HTML注入功能
	enable: true,

	// 页脚自定义 HTML 内容
	customHtml: `<p>© 2024-2026 枫语 | Feng Yu</p>
<p><a href="https://beian.miit.gov.cn/" target="_blank" rel="noopener noreferrer">苏ICP备2021000000号</a></p>`,
};

// 直接编辑 config/FooterConfig.html 文件来添加备案号等自定义内容
