# voiage Go Binding

Go module scaffold for the voiage core API contract.

## Setup

From `bindings/go/`:

```bash
go test ./...
go vet ./...
```

## First workflow

```go
package main

import (
	"fmt"

	voiage "github.com/edithatogo/voiage/bindings/go"
)

func main() {
	evpi, err := voiage.EVPI([][]float64{
		{10.0, 1.0},
		{2.0, 8.0},
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("EVPI: %.1f\n", evpi)
}
```

This prints `EVPI: 3.0` for the simple two-strategy matrix above. The Go
package calls the Rust `voiage_v1_evpi` C ABI; it does not contain an
independent numerical implementation. CI builds the ABI library before the
Go test and vet gates.

## Release and caveats

Go modules are published by pushing semver tags under `bindings/go/v*`. The
release workflow validates the module with `-mod=readonly`, attaches a
versioned source archive to the GitHub release, and lets the Go module proxy
index the tagged release. The README example intentionally stays thin: it
mirrors the Rust-core ownership model rather than re-implementing the VOI
logic in Go.
