import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import polyglot, { sidebarGroup } from "astro-polyglot";
import starlightLinksValidator from "starlight-links-validator";
import starlightLlmsTxt from "starlight-llms-txt";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

export default defineConfig({
  site: "https://edithatogo.github.io/voiage",
  base: "/voiage",
  trailingSlash: "never",

  integrations: [
    starlight({
      title: "voiage",
      description:
        "Cross-domain Value of Information (VOI) analysis library for Rust, Python, Mojo, R, and Julia",
      favicon: "/favicon.ico",

      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/edithatogo/voiage",
        },
      ],

      editLink: {
        baseUrl:
          "https://github.com/edithatogo/voiage/edit/main/docs/astro-site",
      },

      lastUpdated: true,

      plugins: [
        polyglot({
          contentDir: "src/content/docs",
          python: {
            entryPoints: [path.join(repositoryRoot, "voiage")],
            output: "api-reference/generated/python",
            basePath: "/voiage",
            pythonExecutable:
              process.env.STARLIGHT_POLYGLOT_PYTHON ??
              path.join(repositoryRoot, ".venv/bin/python"),
          },
          failFast: true,
        }),
        starlightLinksValidator(),
        starlightLlmsTxt(),
      ],

      sidebar: [
          {
            label: "Start Here",
            items: [
              { link: "/getting-started/", label: "Getting Started" },
              { link: "/introduction/", label: "Introduction" },
              { link: "/faq/", label: "FAQ" },
            ],
        },
        {
          label: "User Guide",
          items: [{ autogenerate: { directory: "user-guide" } }],
        },
        {
          label: "VOI Methods",
          items: [{ autogenerate: { directory: "methods" } }],
        },
        {
          label: "Cross-Domain Usage",
          link: "/cross-domain-usage/",
        },
        {
          label: "API Reference",
          items: [{ autogenerate: { directory: "api-reference" } }],
        },
        sidebarGroup,
        {
          label: "Examples",
          items: [{ autogenerate: { directory: "examples" } }],
        },
        {
          label: "Developer Guide",
          items: [{ autogenerate: { directory: "developer-guide" } }],
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
          label: "Dataset Registry",
          link: "/dataset-registry/",
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
