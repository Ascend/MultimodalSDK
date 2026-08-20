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

- 通过 [Issues](https://gitcode.com/Ascend/MultimodalSDK/issues) 反馈 Bug 或提出功能建议
- 改进或扩展文档
- 审查 Pull Request 并协助其他贡献者
- 传播项目：在博客、社交媒体等渠道分享 Multimodal SDK，或给仓库点个 ⭐

### 开发与测试

1. **启动环境**: 请参考[安装指南](./docs/zh/03_installation_guide/installation_guide.md)部署 Multimodal SDK 开发环境。

2. **构建依赖**: 确保在构建 Multimodal SDK 之前安装了以下依赖：

   | 依赖名称       | 版本建议                     | 获取建议                                                                 |
   | ------------ | ------------------------- | ---------------------------------------------------------------------- |
   | CMake        | 3.14 及以上                  | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y cmake`<br>openEuler：`sudo yum install -y cmake`<br>若版本不符合最低要求，可通过源码编译安装 |
   | Make         | 4.1 及以上                   | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y make`<br>openEuler：`sudo yum install -y make`<br>若版本不符合最低要求，可通过源码编译安装 |
   | GCC          | 9.4 及以上                   | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y build-essential`<br>openEuler：`sudo yum install -y gcc gcc-c++` |
   | SWIG         | 4.3 及以上                  | 建议通过源码安装 |
   | Python       | 最低 3.10；**推荐 3.12** | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y python3 python3-pip python3-dev`<br>openEuler：`sudo yum install -y python3 python3-pip python3-devel`<br>若系统自带版本过低，可从源码编译或安装更高版本 |

3. **Fork 仓库**：在 GitCode 上将本仓库 Fork 到个人账号。

4. **克隆到本地**：

   ```bash
   git clone https://gitcode.com/<your-username>/MultimodalSDK.git
   cd MultimodalSDK
   ```

5. **创建开发分支**：

   ```bash
   git checkout -b <your-branch-name> origin/master
   ```

6. **代码开发**：

   质量符合[开发规范](#dev-rule)和[安全编程指导](#sec-guide)。涉及用户可感知行为的变更，请同步更新文档、样例或 FAQ。

7. **开发构建验证**：

   a. 构建脚本执行，脚本会自动下载并准备编译依赖：

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh

      # 全量编译 + 打包（自动 fetch 依赖）
      bash build_script/build_merge.sh

      构建成功后会在 `output` 目录下生成 `Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run`，可安装此包进行功能验证。

   b. 在提交代码前，请补充测试用例并确保所有测试通过

      测试用例包含大文件（LFS 文件），需要先拉取。

      ```bash
      git lfs pull
      bash build_script/build_merge.sh test
      ```

   c. 如需清理三方依赖、构建中间产物与输出产物，可执行：

      ```bash
      bash build_script/build_merge.sh clean
      ```

8. **执行 pre-commit 检查**

   本地提交代码前请先执行 pre-commit 检查，检查指导参见[pre-commit 本地运行指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

9. **提交 Pull Request**

   - 保持 PR 小规模，一次 PR 只解决一个问题，单个 PR 不超过 1000 行（含测试）代码变更。
   - 及时更新，定期同步上游主分支，及时响应评审意见。
   - 描述清晰，详细描述变更原因和方式，提供测试方法，必要时添加截图或示例。

10. **社区评审**

   如果涉及 patch、头文件宏、API 接口等更新，需提交社区在 SIG 例会进行评审，社区定期例会与活动参见[会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindSeriesSDK)。

## 分支/Tag 命名规则

自研代码仓库

| 分支类型 | 分支名规则 | 示例 | 说明 | tag 名规则 | tag 示例 |
| -------- | ---------- | ---- | ---- | ---------- | -------- |
| 主干&开发 | master | - | - | - | - |
| release | release/<版本号> | release/v26.1.0 | 正式版本 | <版本号>[-beta.<序号>] | v26.1.0，v26.1.0-beta.1 |
| poc | poc/<基线分支>/<描述> | poc/release-v26.1.0/auth-redesign | 后续合入主干 | poc/<基线分支>/<描述>-v<序号> | poc/release-v26.1.0/auth-redesign-v1 |

Fork 开源社区代码仓库

| 分支类型 | 分支名规则 | 示例 | 说明 | tag 名规则 | tag 示例 |
| -------- | ---------- | ---- | ---- | ---------- | -------- |
| 社区分支 | - | v2.1.0 | 不合入代码 | - | - |
| release | release/<社区分支>-<产品版本号> | release/v2.1.0-26.0.0 | 正式版本开发分支 | v<产品版本号>-<社区分支> | v26.0.0-2.1.0 |
| poc | poc/<基线分支>/<描述> | poc/release-v26.1.0/auth-redesign | 后续合入 release 分支 | poc/<基线分支>/<描述>-v<序号> | poc/release-v26.1.0/auth-redesign-v1 |

## 参考

- 开发规范<a id="dev-rule"></a>
  - [《Ascend C++ 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-coding-style-guide.md)
  - [《Ascend Python 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
  - [《Ascend Go 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-go-coding-style-guide.md)
- 安全编程指导<a id="sec-guide"></a>
  - [《Ascend C++ 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-secure-coding-guide.md)
  - [《Ascend Python 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-secure-coding-guide.md)
  - [《Ascend Go 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-go-secure-coding-guide.md)
- [《Ascend 安全编译选项指南（C&C++）》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-secure-compile-guide.md)
- [《Ascend 社区开发者测试贡献指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [《Ascend 开源与第三方软件管理规范》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- 更多社区相关规范，请访问 [Ascend 社区 community](https://gitcode.com/Ascend/community)。
