import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

function loadRustWasm() {
  try {
    return require("../wasm/voiage_wasm.cjs");
  } catch (error) {
    throw new Error(
      "The Rust WebAssembly adapter is unavailable; run `npm run build:wasm` before using @voiage/core.",
      { cause: error },
    );
  }
}

/**
 * Calculate Expected Value of Perfect Information through Rust WebAssembly.
 *
 * @param {number[][]} netBenefits rows are samples, columns are strategies
 * @returns {number}
 */
export function evpi(netBenefits) {
  validateNetBenefits(netBenefits);
  if (netBenefits.length === 0 || netBenefits[0].length <= 1) {
    return 0;
  }
  const values = Float64Array.from(netBenefits.flat());
  return loadRustWasm().evpi(values, netBenefits.length, netBenefits[0].length);
}

/**
 * Validate the canonical EVPI fixture shape.
 *
 * @param {{analysis_type: string, evpi: number}} fixture
 * @returns {boolean}
 */
export function validateEvpiFixture(fixture) {
  return fixture.analysis_type === "evpi" && Number.isFinite(fixture.evpi);
}

function validateNetBenefits(netBenefits) {
  if (!Array.isArray(netBenefits)) {
    throw new TypeError("netBenefits must be an array of rows.");
  }
  if (netBenefits.length === 0) {
    return;
  }
  if (!Array.isArray(netBenefits[0]) || netBenefits[0].length === 0) {
    throw new TypeError("netBenefits must contain non-empty rows.");
  }
  const width = netBenefits[0].length;
  for (const row of netBenefits) {
    if (!Array.isArray(row) || row.length !== width) {
      throw new TypeError("netBenefits rows must have a consistent width.");
    }
    for (const value of row) {
      if (!Number.isFinite(value)) {
        throw new TypeError("netBenefits values must be finite numbers.");
      }
    }
  }
}
