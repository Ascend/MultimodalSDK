# 贡献指南

感谢您考虑为 Ascend Multimodal SDK 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等，甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

## 贡献方式

请先提前了解社区相关规范：

- [签署贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)
- [社区行为准则](https://gitcode.com/Ascend/community/blob/master/docs/contributor/code-of-conduct.md)
- [Issue 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)
- [社区 Issue 处理流程指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-workflow-guidelines.md)
- [Ascend 社区开发者测试贡献指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [Ascend 开源与第三方软件建仓及分支命名指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)
- [Ascend 开源与第三方软件管理规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- [社区安全设计规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)
- [Python代码规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- [Python安全编码指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-secure-coding-guide.md)
- [C++代码规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-coding-style-guide.md)
- [C++安全编码指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-secure-coding-guide.md)
- [PR 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)

您可以通过多种方式支持本项目：

- [领取社区任务](https://www.hiascend.com/developer/activities/details/5dbf59b2dce14f91afb157f5a52f332d#tab0)
- 通过 [Issues](https://gitcode.com/Ascend/MultimodalSDK/issues) 反馈 Bug 或提出功能建议
- 改进或扩展文档
- 审查 Pull Request 并协助其他贡献者
- 传播项目：在博客、社交媒体等渠道分享 Multimodal SDK，或给仓库点个 ⭐

### 开发与测试

1. **Fork 仓库**：在 GitCode 上将本仓库 Fork 到个人账号。
2. **克隆到本地**：

   ```bash
   git clone https://gitcode.com/<your-username>/MultimodalSDK.git
   cd MultimodalSDK
   ```

3. **创建开发分支**：

   ```bash
   git checkout -b <your-branch-name> origin/master
   ```

4. **代码开发**：请遵循代码规范。
5. **开发构建验证**：

   环境准备请参阅[安装指南](docs/zh/03_installation_guide/installation_guide.md)（Atlas 800I A2、Ubuntu 22.04 aarch64、CANN 9.1.0）。clone 后直接执行 `bash build_script/build_merge.sh`，脚本会自动下载并准备全部编译依赖（opensource、makeself、pybind11、googletest 等）：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   # 全量编译 + 打包（自动 fetch 依赖）
   bash build_script/build_merge.sh

   # 运行 AccSDK gtest + MultimodalSDK pytest
   bash build_script/build_merge.sh test

   # 清理三方依赖、构建中间产物与输出产物（无需 CANN 环境）
   bash build_script/build_merge.sh clean
   ```

   子项目也可独立构建：

   ```bash
   # 仅构建 AccSDK
   cd AccSDK/build_script && ./build.sh

   # 仅构建 MultimodalSDK（需 AccSDK 已编译）
   cd MultimodalSDK/build_script && ./build.sh
   ```

6. **本地门禁检查**：请先运行 pre-commit 进行本地检查，确保代码符合规范。
7. **提交 Pull Request**：提交 PR 并等待代码审查。
8. **社区评审**：如果涉及 patch、头文件宏、API 接口等变更，需提交社区评审。
