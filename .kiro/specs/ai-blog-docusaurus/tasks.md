# 实现计划：AI 技术博客（Docusaurus）

## 概述

基于 Docusaurus 3.x 搭建 AI 技术博客，配置中文支持和 GitHub Pages 自动部署。使用 TypeScript 进行配置，Vitest + fast-check 进行测试。

## 任务

- [x] 1. 初始化 Docusaurus 项目并配置基础设置
  - [x] 1.1 使用 Docusaurus classic 模板初始化项目，配置 TypeScript 支持
    - 执行 `npx create-docusaurus@latest . classic --typescript` 或手动创建项目结构
    - 确保 package.json 包含 dev、build、serve 脚本
    - _Requirements: 1.1, 6.1, 6.2, 6.3_
  - [x] 1.2 配置 docusaurus.config.ts 核心设置
    - 设置 i18n defaultLocale 为 zh-Hans
    - 配置站点标题、tagline、favicon
    - 配置 url、baseUrl、organizationName、projectName 用于 GitHub Pages
    - 配置 blog 预设：showReadingTime、blogSidebarTitle、blogSidebarCount
    - _Requirements: 1.2, 1.3, 1.4, 5.9_
  - [x] 1.3 配置导航栏
    - 添加博客（/blog）、归档（/blog/archive）、关于（/about）导航链接
    - 配置 colorMode 支持明暗主题切换
    - _Requirements: 2.3, 4.4_

- [x] 2. 中文排版样式与自定义页面
  - [x] 2.1 创建 src/css/custom.css 中文排版样式
    - 配置中文字体栈（PingFang SC、Hiragino Sans GB、Microsoft YaHei）
    - 设置行高 1.8 和适当的段落间距
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 2.2 创建关于页面 src/pages/about.tsx
    - 创建博客和作者介绍页面
    - _Requirements: 2.3_

- [x] 3. 博客内容与示例文章
  - [x] 3.1 创建 blog/authors.yml 作者信息文件
    - 定义作者名称、头衔、头像等信息
    - _Requirements: 1.3_
  - [x] 3.2 创建 AI 技术介绍示例博客文章
    - 包含完整的前置元数据（title、date、authors、tags）
    - 展示代码块（含语法高亮）、图片引用的使用方式
    - 使用预设 AI 标签：机器学习、深度学习、自然语言处理、计算机视觉、大语言模型
    - _Requirements: 2.4, 2.5, 3.1, 3.2, 3.3_
  - [x] 3.3 安装并配置数学公式支持（remark-math + rehype-katex）
    - 安装 remark-math 和 rehype-katex 插件
    - 在 docusaurus.config.ts 中配置插件
    - 在示例文章中添加数学公式示例
    - _Requirements: 3.2_

- [x] 4. Checkpoint - 本地构建验证
  - 运行 `npm run build` 确保站点构建成功
  - 确保所有页面和文章正确生成，如有问题请询问用户

- [x] 5. GitHub Actions CI/CD 配置
  - [x] 5.1 创建部署工作流 .github/workflows/deploy.yml
    - 配置 push 到 main 分支触发
    - 使用 Node.js 20 LTS 和 npm 缓存
    - 包含 npm ci、npm run build 步骤
    - 使用 actions/upload-pages-artifact 和 actions/deploy-pages 部署
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
  - [x] 5.2 创建 PR 构建验证工作流 .github/workflows/ci.yml
    - 配置 pull_request 到 main 分支触发
    - 只执行构建验证，不触发部署
    - _Requirements: 5.10_

- [x] 6. 测试设置与配置验证测试
  - [x] 6.1 安装测试依赖并配置 Vitest
    - 安装 vitest 和 fast-check
    - 创建 vitest.config.ts 配置文件
    - 在 package.json 中添加 test 脚本
    - _Requirements: 6.3_
  - [ ]* 6.2 编写配置验证单元测试（__tests__/config.test.ts）
    - 验证 i18n 配置为 zh-Hans
    - 验证导航栏包含博客、归档、关于链接
    - 验证 GitHub Pages 相关配置字段存在
    - 验证 package.json 包含 dev、build、serve 脚本
    - _Requirements: 1.2, 2.3, 5.9, 6.3_
  - [ ]* 6.3 编写属性测试：文章排序（__tests__/blog-sorting.prop.ts）
    - **Property 1: 文章列表按时间倒序排列**
    - **Validates: Requirements 2.1**
  - [ ]* 6.4 编写属性测试：标签筛选（__tests__/tag-filtering.prop.ts）
    - **Property 2: 标签筛选返回正确文章**
    - **Validates: Requirements 2.2**

- [x] 7. Final Checkpoint - 确保所有测试通过
  - 运行 `npm test` 确保所有测试通过
  - 运行 `npm run build` 确保构建成功
  - 确保所有测试通过，如有问题请询问用户

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP 交付
- 每个任务引用了具体的需求编号以确保可追溯性
- 属性测试验证通用正确性属性，单元测试验证具体配置示例
- Checkpoint 任务用于增量验证
