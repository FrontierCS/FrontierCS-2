pub mod bypass;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::iter::FromIterator;

pub fn serialize<'a, T, K, V, S>(target: T, ser: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: IntoIterator<Item = (&'a K, &'a V)>,
    K: Serialize + 'a,
    V: Serialize + 'a,
{
    let container: Vec<_> = target.into_iter().collect();
    serde::Serialize::serialize(&container, ser)
}

pub fn deserialize<'de, T, K, V, D>(des: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromIterator<(K, V)>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    let container: Vec<_> = serde::Deserialize::deserialize(des)?;
    Ok(container.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use crate as vectorize;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MyKey {
        one: String,
        two: u16,
        more: Vec<u8>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct MyComplexType {
        #[serde(with = "vectorize")]
        map: HashMap<MyKey, String>,
    }

    #[test]
    fn it_works() -> Result<(), Box<dyn std::error::Error>> {
        let key = MyKey {
            one: "1".into(),
            two: 2,
            more: vec![1, 2, 3],
        };
        let mut map = HashMap::new();
        map.insert(key.clone(), "value".into());
        let instance = MyComplexType { map };
        let serialized = serde_json::to_string(&instance)?;
        println!("JSON: {}", serialized);
        let deserialized: MyComplexType = serde_json::from_str(&serialized)?;
        let expected_value = "value".to_string();
        assert_eq!(deserialized.map.get(&key), Some(&expected_value));
        Ok(())
    }
}
