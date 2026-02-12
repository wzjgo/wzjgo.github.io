import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'AI 技术博客',
  tagline: '探索人工智能的无限可能',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://wzjgo.github.io',
  baseUrl: '/',

  organizationName: 'wzjgo',
  projectName: 'wzjgo.github.io',

  trailingSlash: false,

  onBrokenLinks: 'throw',

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  presets: [
    [
      'classic',
      {
        docs: false,
        blog: {
          showReadingTime: true,
          blogSidebarTitle: '最新文章',
          blogSidebarCount: 'ALL',
          tagsBasePath: 'tags',
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'AI 技术博客',
      items: [
        { to: '/blog', label: '博客', position: 'left' },
        { to: '/blog/archive', label: '归档', position: 'left' },
        { to: '/about', label: '关于', position: 'left' },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '内容',
          items: [
            {
              label: '博客',
              to: '/blog',
            },
            {
              label: '归档',
              to: '/blog/archive',
            },
          ],
        },
        {
          title: '更多',
          items: [
            {
              label: '关于',
              to: '/about',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI 技术博客. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
