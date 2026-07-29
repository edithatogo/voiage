# voiage

`voiage` is the supported Rust facade for the VOIAGE Value of Information
library.

The facade exposes validated contracts through module-qualified namespaces:

- `voiage::domain`
- `voiage::diagnostics`
- `voiage::numerics`
- `voiage::serialization`

The C ABI and Python adapter are deliberately not part of this crate. Additive
facade changes follow the workspace semantic-version policy; binding-specific
resource ownership remains in the corresponding leaf adapter.

```rust
use voiage::domain::SampleMatrix;

let values: SampleMatrix = vec![vec![10.0, 4.0], vec![2.0, 8.0]]
    .try_into()
    .expect("valid sample matrix");
let value = voiage::numerics::evpi(&values).expect("EVPI");
assert_eq!(value, 3.0);
```
