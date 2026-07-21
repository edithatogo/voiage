package voiage

/*
#cgo CFLAGS: -I../../rust/crates/voiage-ffi/include
#cgo LDFLAGS: -L../../rust/target/release -lvoiage_ffi -lm
#include "voiage_v1.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// EVPI calculates Expected Value of Perfect Information through the Rust v1
// C ABI. Rows are samples and columns are strategies.
func EVPI(netBenefits [][]float64) (float64, error) {
	if len(netBenefits) == 0 {
		return 0, nil
	}
	if len(netBenefits[0]) == 0 {
		return 0, errors.New("netBenefits must contain non-empty rows")
	}

	width := len(netBenefits[0])
	values := make([]float64, 0, len(netBenefits)*width)
	for _, row := range netBenefits {
		if len(row) != width {
			return 0, errors.New("netBenefits rows must have a consistent width")
		}
		for _, value := range row {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return 0, errors.New("netBenefits values must be finite numbers")
			}
		}
		values = append(values, row...)
	}

	var result C.double
	status := C.voiage_v1_evpi(
		(*C.double)(unsafe.Pointer(&values[0])),
		C.uint64_t(len(netBenefits)),
		C.uint64_t(width),
		(*C.double)(unsafe.Pointer(&result)),
	)
	if status != C.VOIAGE_V1_STATUS_OK {
		return 0, fmt.Errorf("voiage Rust EVPI ABI failed with status %d", int32(status))
	}
	return float64(result), nil
}
