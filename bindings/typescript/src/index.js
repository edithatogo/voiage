/**
 * Calculate Expected Value of Perfect Information from a net-benefit matrix.
 *
 * @param {number[][]} netBenefits rows are samples, columns are strategies
 * @returns {number}
 */
export function evpi(netBenefits) {
  validateNetBenefits(netBenefits);

  const nSamples = netBenefits.length;
  const nStrategies = netBenefits[0].length;
  if (nSamples === 0) {
    return 0;
  }
  if (nStrategies <= 1) {
    return 0;
  }

  const expectedByStrategy = Array.from({ length: nStrategies }, (_, strategyIndex) =>
    mean(netBenefits.map((row) => row[strategyIndex])),
  );
  const expectedCurrentValue = Math.max(...expectedByStrategy);
  const expectedPerfectInformation = mean(netBenefits.map((row) => Math.max(...row)));

  return Math.max(0, expectedPerfectInformation - expectedCurrentValue);
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
  const width = netBenefits[0].length;
  if (!Array.isArray(netBenefits[0]) || width === 0) {
    throw new TypeError("netBenefits must contain non-empty rows.");
  }
  validateConsistentRows(netBenefits, width);
}

function validateConsistentRows(netBenefits, width) {
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

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}
