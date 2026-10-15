# 发布说明

发布包只包含源码、配置、文档、测试和无真实数据示例。它不包含视频、帧、mask、标签、outputs、snapshots、checkpoint、SAM2 权重、缓存或开发环境。执行 `python scripts/build_release.py` 会重建 `dist/sam2_video_annotator_tool`，生成 tar.gz、zip 和 SHA256 文件，并执行编译、测试及关键 CLI 帮助检查。

注意：当前脚本实际会删除并重建整个 `dist/`，且源码白名单尚未包含 `app.py` 和 `.gitignore`。它是原有 CLI 打包工具，不应用来直接制作当前完整 UI 发布包。本次 GitHub 整理没有运行或修改该脚本，也没有新增重复检查脚本。公开 Git 仓库请使用 [发布审计](GITHUB_RELEASE_AUDIT.md) 中的显式暂存建议，并先人工审阅暂存内容。
