# Ascend Multimodal SDK 贡献指南

感谢您考虑为 Ascend Multimodal SDK 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等，甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

## 贡献方式

您可以通过多种方式支持本项目：

- 通过 [Issues](https://gitcode.com/Ascend/MultimodalSDK/issues) 反馈 Bug 或提出功能建议。
- [领取社区任务](#贡献场景)：浏览[开放的社区任务 Issue](https://gitcode.com/Ascend/MultimodalSDK/issues?state=opened)，在目标 Issue 评论 `/assign` 认领后开发并提交 PR。
- 改进或扩展文档。
- 审查 Pull Request 并协助其他贡献者。
- 传播项目：在博客、社交媒体等渠道分享 Multimodal SDK，或给仓库点个 ⭐。

## 贡献场景

根据您的目标选择合适的路径：

| 场景 | 建议流程 |
| -- | -- |
| **Bug 修复** | 在 [Issues](https://gitcode.com/Ascend/MultimodalSDK/issues) 中搜索是否已有相同问题；若无则新建 Issue，评论 `/assign` 认领后提交 PR。 |
| **功能增强** | 请先通过 Issue 与维护者讨论方案，避免 PR 因方向不一致被拒；确认后再开发并提交 PR。 |
| **文档改进** | 可直接提交 PR，仍建议在 Issue 中说明改动范围或关联已有 Issue。 |
| **领取社区任务** | 从[开放的 Issue 列表](https://gitcode.com/Ascend/MultimodalSDK/issues?state=opened)选择任务，评论 `/assign` 认领，完成后关联对应 Issue 提交 PR。 |
| **协助他人** | 在 Issue 中分享解决思路；若需改代码，可通过 `/assign` 认领并跟进。 |

## 快速开始

完整贡献路径如下：

1. **Fork 仓库**：在 GitCode 上将本仓库 Fork 到个人账号。
2. **克隆到本地**：

   ```bash
   git clone https://gitcode.com/<your-username>/MultimodalSDK.git
   cd MultimodalSDK
   ```

3. **代码开发**：遵循[代码规范](#代码规范)，基于 `master` 创建特性分支。
4. **运行测试**：参见[代码测试](#代码测试)，确保改动通过测试。
5. **本地门禁检查**：提交前运行 pre-commit，参见[本地门禁检查](#本地门禁检查)。
6. **更新文档**（如适用）：参见[文档开发](#文档开发)。
7. **提交 Pull Request**：参见[提交 Pull Request](#提交-pull-request)。

## 代码规范

### Python 代码规范

- 遵循 PEP 8 编码规范
- 使用 4 个空格进行缩进
- 类名使用大驼峰命名法（如 `DataManager`）
- 函数和变量使用小写加下划线命名法（如 `parse_data`）
- 添加必要的类型注解和文档字符串

### C++ 代码规范

- 遵循项目现有的编码风格
- 使用 4 个空格进行缩进
- 类名使用大驼峰命名法
- 函数名使用小驼峰命名法
- 添加必要的注释说明复杂逻辑

## 代码测试

### 运行测试

在提交代码前，请确保所有测试通过。可执行以下命令运行测试用例：

```bash
bash build_script/build_merge.sh test
```

环境准备与编译说明请参阅[安装指南](docs/zh/installation_guide.md)。

### 添加测试

- 为新功能添加相应的单元测试
- 确保测试覆盖主要逻辑分支
- 测试用例应具有良好的可读性和维护性

## 本地门禁检查

项目使用 pre-commit 在提交前自动完成代码校验与格式化。首次贡献前请完成安装与钩子注册，详见社区 [pre-commit 本地运行指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

常用命令：

```bash
pip install pre-commit
pre-commit install
git add .
pre-commit run
```

## 文档开发

### 文档路径

若您的更改影响用户使用方式，请更新相关文档：

- 文档导航入口：[docs/zh/README.md](docs/zh/README.md)
- 常见更新目标：安装指南、快速入门、用户指南、API 参考等（路径见上述导航页）

### 文档规范

- 使用简洁明了的中文表述
- 提供完整、可运行的示例代码
- 包含必要的截图或图表说明（图片资源随发布文档包或 Git LFS 提供，路径以 `docs/zh/figures/` 为准）
- 确保链接有效

### 本地构建文档站点（可选）

```bash
pip install -r requirements-docs.txt
./scripts/mkdocs.sh serve
```

浏览器访问 `http://127.0.0.1:8000` 预览文档。配置见 [`mkdocs.yml`](mkdocs.yml)。

## 提交 Pull Request

### 贡献者许可协议（CLA）

首次向 Ascend 社区提交代码前，需签署贡献者许可协议（CLA）。若 PR 上出现 `ascend-cla/no` 标签，请按社区机器人提示完成签署；签署成功后 PR 将获得 `ascend-cla/yes` 标签。详见社区 [PR 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md) 中的自动化测试说明。

### 提交前检查清单

在提交 Pull Request 之前，请确保：

- [ ] 代码遵循项目的编码规范
- [ ] 添加了必要的测试用例，且 `bash build_script/build_merge.sh test` 已通过
- [ ] 已通过 pre-commit 本地检查
- [ ] 更新了相关文档（如适用）
- [ ] 提交信息清晰明确，使用 `git commit -S` 签名提交
- [ ] 已完成 CLA 签署（PR 上出现 `ascend-cla/yes`）
- [ ] 代码已经过自我审查
- [ ] 已按 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.md) 完成开发自检，包括：关联 Issue 或里程碑、UT/ST 说明、资料与接口变更、PR 规模（未超过 1k 行或已备案）

### 提交流程

1. 创建特性分支，使用 `git commit -S` 签名提交代码。
2. 推送分支到 Fork 仓库。
3. 在 GitCode 创建 Pull Request，按 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.md) 填写描述，并关联对应 Issue。

详细操作（关联 Issue、分配评审人、触发 CI、审核与合入标签、Bot 命令等）请参阅社区 [PR 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)。

提交前本地代码质量检查（pre-commit 安装、运行、常见问题）请参阅社区 [pre-commit 本地运行指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

### Pull Request 最佳实践

- 保持 PR 规模适中，一个 PR 只解决一个问题
- 及时响应审查意见，保持与主分支同步
- 更多规范见社区 [PR 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)

## 社区准则

### 行为准则

我们致力于为所有参与者提供一个友好、安全、包容的环境。参与本项目即表示您同意遵守 [Ascend 社区行为准则](https://gitcode.com/Ascend/community/blob/master/docs/contributor/code-of-conduct.md)，包括：

- 尊重不同的观点和经验
- 接受建设性的批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 沟通渠道

- **Issues**：[报告 Bug、提出功能建议](https://gitcode.com/Ascend/MultimodalSDK/issues)
- **社区任务**：见上文[贡献场景](#贡献场景)中的「领取社区任务」
- **Pull Requests**：[代码审查与讨论](https://gitcode.com/Ascend/MultimodalSDK/merge_requests)

## 许可证

通过向本项目贡献代码，您同意您的贡献将按照 [MulanPSL-2.0](LICENSE.md) 许可证进行授权。

## 致谢

感谢您为 Multimodal SDK 做出的贡献。您的努力使这个项目变得更加强大和用户友好。期待您的参与！

---

如有任何疑问或需要帮助，请随时在 [Issues](https://gitcode.com/Ascend/MultimodalSDK/issues) 中提问，或通过上述社区渠道与我们联系。
