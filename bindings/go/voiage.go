package voiage

import "errors"

// EVPI calculates Expected Value of Perfect Information from a net-benefit matrix.
// Rows are samples and columns are strategies.
func EVPI(netBenefits [][]float64) (float64, error) {
	if len(netBenefits) == 0 {
		return 0, nil
	}
	if len(netBenefits[0]) == 0 {
		return 0, errors.New("netBenefits must contain non-empty rows")
	}

	width := len(netBenefits[0])
	strategySums := make([]float64, width)
	maxSum := 0.0

	for _, row := range netBenefits {
		if len(row) != width {
			return 0, errors.New("netBenefits rows must have a consistent width")
		}

		rowMax := row[0]
		for index, value := range row {
			strategySums[index] += value
			if value > rowMax {
				rowMax = value
			}
		}
		maxSum += rowMax
	}

	if width <= 1 {
		return 0, nil
	}

	nSamples := float64(len(netBenefits))
	maxExpected := strategySums[0] / nSamples
	for _, sum := range strategySums[1:] {
		expected := sum / nSamples
		if expected > maxExpected {
			maxExpected = expected
		}
	}

	evpi := maxSum/nSamples - maxExpected
	if evpi < 0 {
		return 0, nil
	}
	return evpi, nil
}
