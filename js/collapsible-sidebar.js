// 左侧可伸缩边栏功能

(function() {
    'use strict';
    
    // 初始化侧边栏
    function initCollapsibleSidebar() {
        const sidebar = document.querySelector('.collapsible-sidebar');
        const toggle = document.querySelector('.sidebar-toggle');
        const overlay = document.querySelector('.sidebar-overlay');
        
        if (!sidebar || !toggle) return;
        
        // 切换侧边栏状态
        function toggleSidebar() {
            sidebar.classList.toggle('open');
            toggle.classList.toggle('open');
            
            if (overlay) {
                overlay.classList.toggle('show');
            }
            
            // 更新按钮图标
            const icon = toggle.querySelector('.icon');
            if (icon) {
                icon.textContent = sidebar.classList.contains('open') ? '✕' : '☰';
            }
            
            // 保存状态到 localStorage
            localStorage.setItem('sidebarOpen', sidebar.classList.contains('open'));
        }
        
        // 点击切换按钮
        toggle.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleSidebar();
        });
        
        // 点击遮罩关闭
        if (overlay) {
            overlay.addEventListener('click', function() {
                if (sidebar.classList.contains('open')) {
                    toggleSidebar();
                }
            });
        }
        
        // 按 ESC 键关闭
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && sidebar.classList.contains('open')) {
                toggleSidebar();
            }
        });
        
        // 恢复上次的状态
        const savedState = localStorage.getItem('sidebarOpen');
        if (savedState === 'true') {
            sidebar.classList.add('open');
            toggle.classList.add('open');
            if (overlay) {
                overlay.classList.add('show');
            }
            const icon = toggle.querySelector('.icon');
            if (icon) icon.textContent = '✕';
        }
        
        // 监听窗口大小变化
        let resizeTimer;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function() {
                // 在桌面端自动打开侧边栏
                if (window.innerWidth > 1200 && !sidebar.classList.contains('open')) {
                    toggleSidebar();
                }
            }, 250);
        });
        
        // 页面加载完成后，如果是桌面端，自动打开侧边栏
        if (window.innerWidth > 1200 && savedState !== 'false') {
            setTimeout(function() {
                if (!sidebar.classList.contains('open')) {
                    toggleSidebar();
                }
            }, 500);
        }
    }
    
    // 更新系列进度
    function updateSeriesProgress() {
        const progressBars = document.querySelectorAll('.series-progress-bar');
        
        progressBars.forEach(function(bar) {
            const progress = bar.dataset.progress || 0;
            setTimeout(function() {
                bar.style.width = progress + '%';
            }, 300);
        });
    }
    
    // DOM 加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initCollapsibleSidebar();
            updateSeriesProgress();
        });
    } else {
        initCollapsibleSidebar();
        updateSeriesProgress();
    }
    
    // 添加平滑滚动到锚点
    document.addEventListener('click', function(e) {
        if (e.target.matches('a[href^="#"]')) {
            e.preventDefault();
            const target = document.querySelector(e.target.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    });
    
})();