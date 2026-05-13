import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightVersions from "starlight-versions";
import starlightLinksValidator from "starlight-links-validator";
import starlightLlmsTxt from "starlight-llms-txt";
import starlightPolyglot from "starlight-polyglot";

export default defineConfig({
  site: "https://edithatogo.github.io/voiage",
  base: "/voiage",
  outDir: "../../dist",

  trailingSlash: "never",

  integrations: [
    starlight({
      title: "voiage",
      description:
        "Cross-domain Value of Information (VOI) analysis library for Python, R, Julia, TypeScript, Go, Rust, and .NET",
      favicon: "/favicon.ico",

      logo: {
        source: "/img/voiage-logo.svg",
        replacesTitle: false,
      },

      social: {
        github: "https://github.com/edithatogo/voiage",
      },

      editLink: {
        baseUrl:
          "https://github.com/edithatogo/voiage/edit/main/docs/astro-site",
      },

      lastUpdated: true,

      components: {
        ThemeProvider: "starlight-versions/components/ThemeProvider",
        ThemeSelect: "starlight-versions/components/VersionPicker",
      },

      plugins: [
        starlightVersions({
          versions: [
            { slug: "latest", label: "v0.3.x (latest)" },
            { slug: "v0.2", label: "v0.2.x" },
          ],
          defaultVersion: "latest",
        }),

        starlightLinksValidator(),

        starlightLlmsTxt(),

        starlightPolyglot({
          languages: [
            {
              id: "python",
              label: "Python",
              entryPoints: ["voiage"],
              output: "api/python",
            },
            {
              id: "typescript",
              label: "TypeScript",
              entryPoints: ["voiage"],
              output: "api/typescript",
            },
          ],
          defaultLanguage: "python",
        }),
      ],

      sidebar: [
        {
          label: "Start Here",
          items: [
            { slug: "getting-started", label: "Getting Started" },
            { slug: "introduction", label: "Introduction" },
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
          slug: "cross-domain-usage",
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
          slug: "cli-reference",
        },
        {
          label: "Data Structures",
          slug: "data-structures",
        },
        {
          label: "Backends",
          slug: "backends",
        },
        {
          label: "Contributing",
          slug: "contributing",
        },
        {
          label: "Changelog",
          slug: "changelog",
        },
      ],
    }),
  ],
});
