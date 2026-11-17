# vectorize

Converts maps to vecs for serialization.

Usage:

```rust

#[derive(Deserialize, Serialize)]
struct MyStruct {
    ordinary_field: String,
    #[serde(with = "vectorize")]
    no_string_key_map: HashMap<(u8, bool, String), String>,
}

```
