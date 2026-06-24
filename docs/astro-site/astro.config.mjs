import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightLlmsTxt from "starlight-llms-txt";

export default defineConfig({
  site: "https://edithatogo.github.io/voiage",
  base: "/voiage",
  trailingSlash: "never",

  integrations: [
    starlight({
      title: "voiage",
      description:
        "Cross-domain Value of Information (VOI) analysis library for Python, R, Julia, TypeScript, Go, Rust, and .NET",
      favicon: "/favicon.ico",

      social: {
        github: "https://github.com/edithatogo/voiage",
      },

      editLink: {
        baseUrl:
          "https://github.com/edithatogo/voiage/edit/main/docs/astro-site",
      },

      lastUpdated: true,

      plugins: [
        starlightLlmsTxt(),

      ],

      sidebar: [
        {
          label: "Start Here",
          items: [
            { link: "/getting-started/", label: "Getting Started" },
            { link: "/introduction/", label: "Introduction" },
          ],
        },
        {
          label: "User Guide",
          autogenerate: { directory: "user-guide" },
        },
        {
          label: "VOI Methods",
          autogenerate: { directory: "methods" },
        },
        {
          label: "Cross-Domain Usage",
          link: "/cross-domain-usage/",
        },
        {
          label: "API Reference",
          autogenerate: { directory: "api-reference" },
        },
        {
          label: "Examples",
          autogenerate: { directory: "examples" },
        },
        {
          label: "Developer Guide",
          autogenerate: { directory: "developer-guide" },
        },
        {
          label: "CLI Reference",
          link: "/cli-reference/",
        },
        {
          label: "Data Structures",
          link: "/data-structures/",
        },
        {
          label: "Backends",
          link: "/backends/",
        },
        {
          label: "Contributing",
          link: "/contributing/",
        },
        {
          label: "Changelog",
          link: "/changelog/",
        },
      ],
    }),
  ],
});
