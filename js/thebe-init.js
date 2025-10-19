(function () {
    function tagExecutableCells() {
        const codeBlocks = document.querySelectorAll("pre code.language-python, pre code.language-julia, pre code.language-r");
        codeBlocks.forEach(function (code) {
            if (code.dataset.thebeExecutable === "false") {
                return;
            }
            const pre = code.parentElement;
            if (!pre) {
                return;
            }
            pre.classList.add("thebe-cell");
            pre.setAttribute("data-thebe-executable", "true");
            pre.setAttribute("data-language", code.className.replace("language-", ""));
        });
    }

    function waitForThebe(attempts) {
        if (window.thebe && typeof window.thebe.bootstrap === "function") {
            window.thebe.bootstrap();
            return;
        }
        if (attempts > 20) {
            console.warn("Thebe bootstrap 超时，请检查 Thebe 资源是否加载成功。");
            return;
        }
        setTimeout(function () {
            waitForThebe((attempts || 0) + 1);
        }, 300);
    }

    document.addEventListener("DOMContentLoaded", function () {
        if (!document.querySelector("[data-thebe-executable]")) {
            tagExecutableCells();
        }
        if (document.querySelector(".thebe-cell")) {
            waitForThebe(0);
        }
    });
})();
