import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { evpi, validateEvpiFixture } from "../src/index.js";

test("evpi calculates the canonical simple matrix", () => {
  assert.equal(
    evpi([
      [10, 1],
      [2, 8],
    ]),
    3,
  );
});

test("evpi validates row width", () => {
  assert.throws(() => evpi([[1], [1, 2]]), /consistent width/);
});

test("canonical EVPI fixture is recognized", async () => {
  const fixture = JSON.parse(
    await readFile(
      new URL("../../../specs/core-api/fixtures/v1/normative/evpi.json", import.meta.url),
      "utf8",
    ),
  );

  assert.equal(validateEvpiFixture(fixture), true);
});
