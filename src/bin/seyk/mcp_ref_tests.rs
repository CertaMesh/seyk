use super::*;

fn sample() -> Selection {
    Selection {
        resource: "https://example.org/fixture".parse().unwrap(),
        locator: ResourceLocator::WholeResource,
        metadata: ResourceMetadata {
            modality: Modality::Text,
            size_bytes: 0,
            modified_ns: None,
            media_type: None,
            content_hash: "00".repeat(32),
        },
    }
}

#[test]
fn literal_vectors_cover_codec_and_separate_selection_validation() {
    let vectors: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mcp-refs.json"
    )))
    .unwrap();
    for v in vectors.as_array().unwrap() {
        #[cfg(not(unix))]
        if v["omit_resource"] == true
            || v["selection"]["resource"].as_str().unwrap().contains("%FF")
        {
            continue;
        }
        let s: Selection = serde_json::from_value(v["selection"].clone()).unwrap();
        let encoded = encode(&s, v["path"].as_str()).unwrap();
        assert_eq!(encoded, v["ref"].as_str().unwrap(), "{}", v["name"]);
        let binary = URL_SAFE_NO_PAD
            .decode(encoded.strip_prefix(PREFIX).unwrap())
            .unwrap();
        let hex: String = binary.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, v["hex"].as_str().unwrap());
        let decoded = decode(&encoded, v["path"].as_str()).unwrap();
        assert!(decoded == s);
        // Some(0) has a canonical codec representation, but line zero cannot select evidence.
        assert_eq!(
            validate(&decoded).is_ok(),
            v["valid_selection"].as_bool().unwrap()
        );
    }
}

#[test]
fn missing_path_and_edited_display_fields_have_explicit_identity_semantics() {
    let mut s = sample();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("a");
    s.resource = ResourceId::from_path(&path).unwrap();
    let path = path.to_str().unwrap();
    let reference = encode(&s, Some(path)).unwrap();
    assert!(decode(&reference, None).is_err());
    assert!(decode(&reference, Some("relative")).is_err());
    assert!(decode(&reference, Some(path)).unwrap() == s);
    let explicit = encode(&s, None).unwrap();
    assert!(decode(&explicit, Some("/edited/ignored")).unwrap() == s);
}

#[test]
fn rejects_noncanonical_truncated_unknown_and_unbounded_references() {
    let good = encode(&sample(), None).unwrap();
    let bytes = URL_SAFE_NO_PAD
        .decode(good.strip_prefix(PREFIX).unwrap())
        .unwrap();
    let wrap = |bytes: &[u8]| format!("{PREFIX}{}", URL_SAFE_NO_PAD.encode(bytes));
    for n in 0..bytes.len() {
        assert!(decode(&wrap(&bytes[..n]), None).is_err());
    }
    for (index, value) in [(0, 0), (2, 2), (3, 2)] {
        let mut b = bytes.clone();
        b[index] = value;
        assert!(decode(&wrap(&b), None).is_err());
    }
    let mut b = bytes.clone();
    b.push(0);
    assert!(decode(&wrap(&b), None).is_err());
    assert!(decode(&(good.clone() + "="), None).is_err());
    assert!(decode(&(good.clone() + "\n"), None).is_err());
    assert!(decode(&good.replace(PREFIX, "seyk-ref-v2:"), None).is_err());
    assert!(decode(&"x".repeat(MAX_REF + 1), None).is_err());
    assert!(Reader(&[0x80, 0]).var().is_err());
    assert!(Reader(&[0xff; 10]).var().is_err());
    assert!(Reader(&[0x80, 0x80, 0x80, 0x80, 0x10]).u32().is_err());
    assert!(Reader(&[2]).optional().is_err());
    assert!(Reader(&[1, 0xff]).string(1).is_err());
    assert!(Reader(&[2, b'a', b'b']).string(1).is_err());
    let mut s = sample();
    for hash in ["00".repeat(31), "AA".repeat(32), "gg".repeat(32)] {
        s.metadata.content_hash = hash;
        assert!(encode(&s, None).is_err());
    }
}

#[test]
fn all_varint_signed_and_string_boundaries_are_lossless() {
    for n in [0, 1, 127, 128, 16383, 16384, u32::MAX as u64, u64::MAX] {
        let mut b = Vec::new();
        var(&mut b, n);
        assert_eq!(Reader(&b).var().unwrap(), n);
    }
    for n in [None, Some(0), Some(-1), Some(i64::MIN), Some(i64::MAX)] {
        let mut s = sample();
        s.metadata.modified_ns = n;
        assert!(decode(&encode(&s, None).unwrap(), None).unwrap() == s);
    }
    let mut s = sample();
    s.metadata.media_type = Some("a".repeat(1024));
    assert!(encode(&s, None).is_ok());
    s.metadata.media_type = Some("a".repeat(1025));
    assert!(encode(&s, None).is_err());
    assert_eq!(MAX_REF, 87_394);
}

#[test]
fn rejects_unknown_tags_noncanonical_media_and_size_boundary_inputs() {
    // Minimal omitted-URI literal from the independently specified seed vector.
    let mut seed = vec![0x53, 0x52, 1, 1, 5, 1, 0, 0, 0];
    seed.extend([0; 32]);
    let wrap = |b: &[u8]| format!("{PREFIX}{}", URL_SAFE_NO_PAD.encode(b));
    let path = if cfg!(windows) {
        "C:/fixture/a"
    } else {
        "/fixture/a"
    };
    assert!(decode(&wrap(&seed), Some(path)).is_ok());
    for index in [4, 5, 7, 8] {
        let mut b = seed.clone();
        b[index] = 255;
        assert!(decode(&wrap(&b), Some(path)).is_err());
    }
    let mut media = seed.clone();
    media.splice(
        8..9,
        [2, 15]
            .into_iter()
            .chain(b"application/pdf".iter().copied()),
    );
    assert!(decode(&wrap(&media), Some(path)).is_err());
    for n in [MAX_BYTES, MAX_BYTES + 1] {
        assert!(decode(&wrap(&vec![0; n]), Some(path)).is_err());
    }
    let mut s = sample();
    s.locator = ResourceLocator::PageText {
        page: 1,
        byte_start: 0,
        byte_end: 128,
        text_version: "x".repeat(4096),
        page_text_hash: "00".repeat(32),
    };
    assert!(encode(&s, None).is_ok());
    if let ResourceLocator::PageText { text_version, .. } = &mut s.locator {
        text_version.push('x');
    }
    assert!(encode(&s, None).is_err());
    s.resource = format!("https://example.org/{}", "a".repeat(MAX_PATH))
        .parse()
        .unwrap();
    assert!(encode(&s, None).is_err());
}
