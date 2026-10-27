use super::*;

fn hit() -> Value {
    json!({"resource":"https://example.org/fixture","locator":{"kind":"text","byte_start":0,"byte_end":2,"line_start":1,"line_end":1},"metadata":{"modality":"text","size_bytes":2,"modified_ns":0,"media_type":"text/plain","content_hash":"00".repeat(32)},"path":"/display/文","lines":"1","score":0.123,"matched":["meaning","words","name"],"match":{},"matched_query":"fixture","preview":"文"})
}

#[test]
fn response_conversion_preserves_existing_f32_wire_decimals() {
    #[derive(serde::Serialize)]
    struct Score {
        score: f32,
    }
    let score = Score { score: 0.123 };
    let old = serde_json::to_string(&score).unwrap();
    assert_eq!(
        serde_json::to_string(&response_value(&score).unwrap()).unwrap(),
        old
    );
}

#[test]
fn strict_selection_table_never_falls_back_from_malformed_full_fields() {
    let full = hit();
    let s: Selection = serde_json::from_value(full.clone()).unwrap();
    let encoded = reference::encode(&s, None).unwrap();
    let slim = json!({"ref":encoded,"path":"/ignored","lines":"999-1000"});
    for input in [
        json!({"hit":full}),
        json!({"hit":slim}),
        json!({"ref":encoded}),
        json!({"ref":encoded,"path":"/ignored"}),
    ] {
        assert!(selection(&parameters(input).unwrap()).unwrap() == s);
    }
    let mut mixed = full.clone();
    mixed["ref"] = encoded.clone().into();
    assert!(selection(&mixed).unwrap() == s);
    mixed["locator"]["byte_end"] = 3.into();
    assert!(selection(&mixed).is_err());
    for input in [
        Value::Null,
        json!({}),
        json!({"hit":null}),
        json!({"ref":null}),
        json!({"ref":""}),
        json!({"ref":encoded,"hit":full}),
        json!({"hit":full,"path":"/ignored"}),
        json!({"ref":encoded,"extra":true}),
    ] {
        assert!(parameters(input).is_err());
    }
    for malformed in [
        json!({"ref":encoded,"resource":full["resource"]}),
        json!({"ref":encoded,"resource":null,"locator":full["locator"],"metadata":full["metadata"]}),
        json!({"ref":encoded,"path":null}),
        json!({"ref":null,"resource":full["resource"],"locator":full["locator"],"metadata":full["metadata"]}),
    ] {
        assert!(selection(&malformed).is_err());
    }
    assert!(serde_json::from_value::<Detail>(Value::Null).is_err());
    assert!(serde_json::from_value::<Detail>(json!("other")).is_err());
    assert_eq!(selection_schema()["oneOf"].as_array().unwrap().len(), 2);
}

#[test]
fn slim_full_keep_ranked_evidence_and_safety_but_differ_in_detail() {
    let response = json!({"schema_version":11,"hits":[hit(),hit()],"kind_counts":{"text":2},"degraded":{"mode":"lexical_only","reason":"model_loading"},"index":{"state":"pass_failed","error":"fixture"},"index_warnings":{"skipped_resources":1,"unindexed_pages":2},"index_progress":[],"skipped":[],"page_diagnostics":[],"took_ms":10});
    let slim = project(response.clone(), Detail::Slim).unwrap();
    let full = project(response.clone(), Detail::Full).unwrap();
    for k in ["kind_counts", "degraded", "index", "index_warnings"] {
        assert_eq!(slim[k], response[k]);
    }
    for i in 0..2 {
        for k in ["score", "preview", "matched", "path", "lines"] {
            assert_eq!(slim["hits"][i][k], response["hits"][i][k]);
        }
        for k in ["resource", "locator", "metadata", "match", "matched_query"] {
            assert!(slim["hits"][i].get(k).is_none());
            assert_eq!(full["hits"][i][k], response["hits"][i][k]);
        }
        assert!(selection(&slim["hits"][i]).unwrap() == selection(&full["hits"][i]).unwrap());
    }
    for k in ["index_progress", "skipped", "page_diagnostics", "took_ms"] {
        assert!(slim.get(k).is_none());
        assert_eq!(full[k], response[k]);
    }
    assert_eq!(
        project(json!({"schema_version":11,"hits":[]}), Detail::Slim).unwrap(),
        json!({"schema_version":11,"hits":[]})
    );
}

#[test]
fn query_warning_counts_are_distinct_uncapped_and_confined_before_aggregation() {
    let a: seyk::ResourceId = "https://example.org/a".parse().unwrap();
    let b: seyk::ResourceId = "https://example.org/b".parse().unwrap();
    let mut sync = seyk::SyncStats::default();
    for resource in [&a, &a, &b] {
        sync.skipped.push(seyk::SkippedResource {
            code: Default::default(),
            resource: resource.clone(),
            path: None,
            reason: "fixture".into(),
        });
    }
    for page in 1..=200 {
        for resource in [&a, &a, &b] {
            sync.page_diagnostics.push(seyk::PageDiagnostic {
                code: Default::default(),
                resource: resource.clone(),
                page,
                reason: "fixture".into(),
            });
        }
    }
    assert_eq!(
        query_warnings(&sync, |r| r == &a).unwrap(),
        IndexWarnings {
            skipped_resources: 1,
            unindexed_pages: 200
        }
    );
    assert!(query_warnings(&sync, |_| false).is_none());
    assert!(query_warnings(&seyk::SyncStats::default(), |_| true).is_none());
}
