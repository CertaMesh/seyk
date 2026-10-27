//! Pure MCP presentation/selection boundary, also used by offline accounting.
#[path = "mcp_ref.rs"]
pub mod reference;

use anyhow::{Result, bail, ensure};
pub use reference::Selection;

/// Preserve the existing serializer's f32 decimal representation before projection.
pub fn response_value(response: &impl Serialize) -> Result<Value> {
    Ok(serde_json::from_str(&serde_json::to_string(response)?)?)
}
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Detail {
    #[default]
    Slim,
    Full,
}

pub fn selection(hit: &Value) -> Result<Selection> {
    let object = hit
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("hit must be an object"))?;
    let count = ["resource", "locator", "metadata"]
        .iter()
        .filter(|k| object.contains_key(**k))
        .count();
    ensure!(count == 0 || count == 3, "partial selected descriptor");
    let full: Option<Selection> = if count == 3 {
        Some(serde_json::from_value(hit.clone())?)
    } else {
        None
    };
    let decoded = if let Some(value) = object.get("ref") {
        let value = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("ref must be a string"))?;
        let path = object
            .get("path")
            .map(|p| {
                p.as_str()
                    .ok_or_else(|| anyhow::anyhow!("path must be a string"))
            })
            .transpose()?;
        if let Some(path) = path {
            ensure!(path.len() <= reference::MAX_PATH, "path exceeds bound");
        }
        Some(reference::decode(value, path)?)
    } else {
        None
    };
    let s = match (full, decoded) {
        (Some(a), Some(b)) => {
            ensure!(a == b, "ref and full descriptor disagree");
            a
        }
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => bail!("a ref or complete resource, locator and metadata is required"),
    };
    reference::validate(&s)?;
    Ok(s)
}

pub fn parameters(value: Value) -> Result<Value> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("selection must be an object"))?;
    let hit = if object.contains_key("hit") {
        ensure!(
            object.len() == 1,
            "hit cannot be combined with ref, path or other parameters"
        );
        value["hit"].clone()
    } else {
        ensure!(
            object.contains_key("ref") && object.keys().all(|k| k == "ref" || k == "path"),
            "expected hit or ref with optional path"
        );
        value
    };
    // Validate the entire decision table before the handler can read a source.
    selection(&hit)?;
    Ok(hit)
}

pub fn selection_schema() -> Value {
    let reference = json!({"type":"string","minLength":13,"maxLength":reference::MAX_REF,"pattern":"^seyk-ref-v1:[A-Za-z0-9_-]+$"});
    let path = json!({"type":"string","maxLength":reference::MAX_PATH});
    let full_properties = json!({"resource":{"type":"string"},"locator":{"type":"object"},"metadata":{"type":"object"},"ref":reference,"path":path});
    let full = json!({"type":"object","required":["resource","locator","metadata"],"properties":full_properties});
    let slim = json!({"type":"object","required":["ref"],"properties":{"ref":reference,"path":path},"not":{"anyOf":[{"required":["resource"]},{"required":["locator"]},{"required":["metadata"]}]}});
    json!({"type":"object","description":"An omitted-URI ref requires path. A ref alone then fails. An explicit-URI ref ignores edited display path.","examples":[{"ref":"seyk-ref-v1:U1IBAQUBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","path":"/fixture/a"},{"ref":"seyk-ref-v1:U1IBABFmaWxlOi8vL2ZpeHR1cmUvYQUBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","path":"/edited/ignored"}],"oneOf":[
        {"required":["hit"],"properties":{"hit":{"oneOf":[full,slim]}},"additionalProperties":false},
        {"required":["ref"],"properties":{"ref":reference,"path":path},"additionalProperties":false}
    ]})
}

/// Input is the current full response. Errors never silently remove a hit.
pub fn project(mut response: Value, detail: Detail) -> Result<Value> {
    let hits = response["hits"]
        .as_array_mut()
        .ok_or_else(|| anyhow::anyhow!("missing hits"))?;
    for hit in hits {
        let s: Selection = serde_json::from_value(hit.clone())?;
        reference::validate(&s)?;
        let encoded = reference::encode(&s, hit["path"].as_str())?;
        hit["ref"] = encoded.into();
        if let seyk::ResourceLocator::Time { start_ms, end_ms } = s.locator {
            hit["time"] = json!({"start_ms":start_ms,"end_ms":end_ms});
        }
        if matches!(detail, Detail::Slim) {
            hit.as_object_mut().unwrap().retain(|k, _| {
                matches!(
                    k.as_str(),
                    "ref"
                        | "path"
                        | "lines"
                        | "page"
                        | "time"
                        | "heading"
                        | "score"
                        | "preview"
                        | "matched"
                )
            });
        }
    }
    if matches!(detail, Detail::Slim) {
        response.as_object_mut().unwrap().retain(|k, _| {
            matches!(
                k.as_str(),
                "schema_version" | "hits" | "kind_counts" | "degraded" | "index" | "index_warnings"
            )
        });
    }
    Ok(response)
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct IndexWarnings {
    pub skipped_resources: usize,
    pub unindexed_pages: usize,
}

pub fn query_warnings(
    sync: &seyk::SyncStats,
    mut allowed: impl FnMut(&seyk::ResourceId) -> bool,
) -> Option<IndexWarnings> {
    use std::collections::BTreeSet;
    let skipped: BTreeSet<_> = sync
        .skipped
        .iter()
        .filter(|s| allowed(&s.resource))
        .map(|s| &s.resource)
        .collect();
    let pages: BTreeSet<_> = sync
        .page_diagnostics
        .iter()
        .filter(|s| allowed(&s.resource))
        .map(|s| (&s.resource, s.page))
        .collect();
    (!skipped.is_empty() || !pages.is_empty()).then_some(IndexWarnings {
        skipped_resources: skipped.len(),
        unindexed_pages: pages.len(),
    })
}

#[cfg(test)]
#[path = "mcp_wire_tests.rs"]
mod tests;
