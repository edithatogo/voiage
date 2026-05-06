# @voiage/core

TypeScript bindings for the voiage core API contract.

This package currently exposes the deterministic EVPI core helper and fixture
validation used by the polyglot CI scaffold. It is intentionally small until the
language-neutral core API reaches a stable v1 binding contract.

## Setup

From `bindings/typescript/`:

```bash
npm install
npm run check
```

## First workflow

```js
import { evpi, validateEvpiFixture } from "@voiage/core";

const evpiValue = evpi([
  [10, 1],
  [2, 8],
]);

console.log(`EVPI: ${evpiValue.toFixed(1)}`);

console.log(
  validateEvpiFixture({
    analysis_type: "evpi",
    evpi: evpiValue,
  }),
);
```

This prints `EVPI: 3.0` and `true` for the canonical two-strategy matrix and
fixture shape.

## Release and caveats

Release tags follow the `typescript-vX.Y.Z` pattern. The release workflow
rewrites the package version from that tag before packing, publishes to npm
with provenance, and attaches a source archive to the GitHub release for the
tag. The binding stays intentionally small because the Rust core owns the
calculation policy; the TypeScript layer is a thin adapter and fixture checker.
