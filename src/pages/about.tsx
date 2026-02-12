import type { ReactNode } from 'react';
import Layout from '@theme/Layout';

export default function About(): ReactNode {
    return (
        <Layout title="关于" description="关于 AI 技术博客">
            <main
                style={{
                    padding: '2rem',
                    maxWidth: '800px',
                    margin: '0 auto',
                }}
            >
                <h1>关于本博客</h1>
                <p>
                    欢迎来到 AI 技术博客！本博客专注于人工智能领域的技术分享与探索，
                    旨在以通俗易懂的方式介绍 AI 前沿技术和实践经验。
                </p>

                <h2>内容方向</h2>
                <p>本博客主要涵盖以下技术领域：</p>
                <ul>
                    <li><strong>机器学习</strong> — 算法原理、模型训练与优化</li>
                    <li><strong>深度学习</strong> — 神经网络架构与应用实践</li>
                    <li><strong>自然语言处理</strong> — 文本理解、生成与对话系统</li>
                    <li><strong>计算机视觉</strong> — 图像识别、目标检测与视觉理解</li>
                    <li><strong>大语言模型</strong> — LLM 技术原理与应用开发</li>
                </ul>

                <h2>关于作者</h2>
                <p>
                    一名热爱 AI 技术的开发者，希望通过这个博客记录学习心得，
                    与更多志同道合的朋友交流分享。
                </p>
            </main>
        </Layout>
    );
}
