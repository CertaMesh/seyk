//! Offline payload accounting through the production MCP projector and ref codec.
#[path = "../src/bin/seyk/mcp_wire.rs"]
pub mod wire;

use anyhow::{Result, ensure};
use base64::Engine;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn count(s: &str) -> Value {
    json!({"chars":s.chars().count(),"bytes":s.len()})
}
fn percentile(values: &mut [usize], p: usize) -> Option<usize> {
    values.sort_unstable();
    (!values.is_empty()).then(|| values[(values.len() * p).div_ceil(100).saturating_sub(1)])
}

fn summarize(calls: &[Value]) -> Value {
    let mut distributions = BTreeMap::new();
    for field in [
        "as_recorded_v10",
        "normalized_v10",
        "v11_full",
        "v11_slim",
        "wrapped_v10",
        "wrapped_v11_full",
        "wrapped_v11_slim",
    ] {
        let mut chars: Vec<_> = calls
            .iter()
            .map(|v| v[field]["chars"].as_u64().unwrap() as usize)
            .collect();
        let mut bytes: Vec<_> = calls
            .iter()
            .map(|v| v[field]["bytes"].as_u64().unwrap() as usize)
            .collect();
        let sum_chars: usize = chars.iter().sum();
        let sum_bytes: usize = bytes.iter().sum();
        distributions.insert(field, json!({"sum_chars":sum_chars,"sum_bytes":sum_bytes,"median_chars":percentile(&mut chars,50),"p90_chars":percentile(&mut chars,90),"median_bytes":percentile(&mut bytes,50),"p90_bytes":percentile(&mut bytes,90)}));
    }
    let mut ratios: Vec<_> = calls
        .iter()
        .map(|v| v["paired_char_ratio"].as_f64().unwrap())
        .collect();
    ratios.sort_by(f64::total_cmp);
    let rank = |p: usize| {
        (!ratios.is_empty()).then(|| ratios[(ratios.len() * p).div_ceil(100).saturating_sub(1)])
    };
    let total = |field: &str| {
        calls
            .iter()
            .map(|v| v[field]["chars"].as_u64().unwrap())
            .sum::<u64>()
    };
    json!({"n":calls.len(),"distributions":distributions,"median_paired_char_ratio":rank(50),"p90_paired_char_ratio":rank(90),"total_char_ratio":if total("v11_slim")>0 {Some(total("normalized_v10") as f64/total("v11_slim") as f64)} else {None}})
}

type TaskKey = (String, String, String, String);
fn task_key(entry: &Value) -> TaskKey {
    (
        entry["agent"].as_str().unwrap().into(),
        entry["arm"].as_str().unwrap().into(),
        entry["cond"].as_str().unwrap().into(),
        entry["task"].as_str().unwrap().into(),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: mcp_payload_sizes MANIFEST EXPECTED_SHA256"
    );
    let bytes = std::fs::read(&args[1])?;
    ensure!(digest(&bytes) == args[2], "manifest hash mismatch");
    let manifest: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest["format"] == "seyk-payload-manifest-v1",
        "unsupported manifest"
    );
    for source in manifest["sources"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing sources"))?
    {
        let raw = std::fs::read(
            source["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing path"))?,
        )?;
        ensure!(
            source["sha256"].as_str() == Some(digest(&raw).as_str()),
            "input file changed since manifest creation"
        );
    }
    let mut calls = Vec::new();
    let mut failed = 0;
    let mut malformed = 0;
    let mut unrepresentable = 0;
    let mut warning_unknown = 0;
    let mut references: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut tasks: BTreeMap<TaskKey, (usize, i64, i64)> = BTreeMap::new();
    for entry in manifest["task_records"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing task records"))?
    {
        tasks.entry(task_key(entry)).or_default();
    }
    for entry in manifest["calls"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing calls"))?
    {
        if entry["is_error"] == true {
            failed += 1;
            continue;
        }
        let raw = entry["raw"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing raw payload"))?;
        ensure!(
            entry["raw_sha256"].as_str() == Some(digest(raw.as_bytes()).as_str()),
            "payload hash mismatch"
        );
        let Ok(old) = serde_json::from_str::<Value>(raw) else {
            malformed += 1;
            continue;
        };
        let Some(hits) = old["hits"].as_array() else {
            malformed += 1;
            continue;
        };
        let normalized = serde_json::to_string(&old)?;
        let mut new = old.clone();
        new["schema_version"] = 11.into();
        // Old wire diagnostics omit skipped identities; do not invent distinct counts.
        let warnings_known = entry.get("query_sync").is_some()
            || (old["skipped"].as_array().is_some_and(|a| a.is_empty())
                && old["page_diagnostics_total"].as_u64() == Some(0));
        if let Some(sync) = entry.get("query_sync") {
            let sync: seyk::SyncStats = serde_json::from_value(sync.clone())?;
            if let Some(warnings) = wire::query_warnings(&sync, |_| true) {
                new["index_warnings"] = serde_json::to_value(warnings)?;
            }
        }
        if !warnings_known {
            warning_unknown += 1;
        }
        let (Ok(full), Ok(slim)) = (
            wire::project(new.clone(), wire::Detail::Full),
            wire::project(new, wire::Detail::Slim),
        ) else {
            unrepresentable += 1;
            continue;
        };
        let full_string = serde_json::to_string(&full)?;
        let slim_string = serde_json::to_string(&slim)?;
        // ContentBlock::json is the same compact JSON text, then a text block.
        let old_wrapped = serde_json::to_string(&vec![rmcp::model::ContentBlock::json(&old)?])?;
        let full_wrapped = serde_json::to_string(&vec![rmcp::model::ContentBlock::json(&full)?])?;
        let slim_wrapped = serde_json::to_string(&vec![rmcp::model::ContentBlock::json(&slim)?])?;
        for hit in full["hits"].as_array().unwrap() {
            let reference = hit["ref"].as_str().unwrap();
            let encoded = reference.strip_prefix(wire::reference::PREFIX).unwrap();
            let binary = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(encoded)?;
            let key = format!(
                "{}:{}",
                hit["locator"]["kind"].as_str().unwrap(),
                if binary[3] == 1 { "path" } else { "uri" }
            );
            references.entry(key).or_default().push(json!({"binary_bytes":binary.len(),"value_chars":reference.len(),"member_chars":serde_json::to_string(&json!({"ref":reference}))?.len()-2}));
        }
        let task = tasks.entry(task_key(entry)).or_default();
        task.0 += 1;
        task.1 += normalized.chars().count() as i64 - slim_string.chars().count() as i64;
        task.2 += normalized.len() as i64 - slim_string.len() as i64;
        calls.push(json!({"agent":entry["agent"],"arm":entry["arm"],"cond":entry["cond"],"hit_count":hits.len(),"as_recorded_v10":count(raw),"normalized_v10":count(&normalized),"v11_full":count(&full_string),"v11_slim":count(&slim_string),"wrapped_v10":count(&old_wrapped),"wrapped_v11_full":count(&full_wrapped),"wrapped_v11_slim":count(&slim_wrapped),"query_warnings_known":warnings_known,"paired_char_ratio":normalized.chars().count() as f64/slim_string.chars().count().max(1) as f64}));
    }
    let all = summarize(&calls);
    let ten_hit = summarize(
        &calls
            .iter()
            .filter(|v| v["hit_count"] == 10)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for call in &calls {
        groups
            .entry(format!(
                "{}:{}:{}",
                call["agent"].as_str().unwrap(),
                call["arm"].as_str().unwrap(),
                call["cond"].as_str().unwrap()
            ))
            .or_default()
            .push(call.clone());
    }
    let groups: BTreeMap<_, _> = groups
        .into_iter()
        .map(|(key, values)| (key, summarize(&values)))
        .collect();
    let reference_costs: BTreeMap<_,_> = references.into_iter().map(|(kind,values)| {
        let mut columns=BTreeMap::new();
        for field in ["binary_bytes","value_chars","member_chars"] {
            let mut sizes:Vec<_>=values.iter().map(|v|v[field].as_u64().unwrap() as usize).collect();
            columns.insert(field,json!({"n":sizes.len(),"min":sizes.iter().min(),"max":sizes.iter().max(),"median":percentile(&mut sizes,50),"p90":percentile(&mut sizes,90)}));
        }
        (kind,columns)
    }).collect();
    // Never print original task names, paths, previews, prompts or errors.
    let projections: Vec<_>=tasks.into_iter().map(|((agent,arm,cond,_),(searches,chars,bytes))|json!({"agent":agent,"arm":arm,"cond":cond,"searches":searches,"projected_char_delta":chars,"projected_byte_delta":bytes})).collect();
    let claude_calls = calls.iter().filter(|v| v["agent"] == "claude").count();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"manifest_sha256":args[2],"build":env!("SEYK_BUILD_VERSION"),"runtime":"Rust / serde_json compact UTF-8; no tokenizer","percentile":"nearest-rank ceil(p*n)","synthetic":manifest["synthetic"],"tasks_in_manifest":manifest["tasks"],"accepted_calls":calls.len(),"tool_errors":failed,"malformed_payloads":malformed,"unrepresentable_descriptors":unrepresentable,"unknown_query_warning_inputs":warning_unknown,"historical_claude_calls_expected":manifest["historical_claude_calls_expected"],"accepted_claude_calls":claude_calls,"all":all,"ten_hit":ten_hit,"groups":groups,"calls":calls,"reference_cost_by_kind":reference_costs,"per_task_projections":projections,"limits":"Known-input projection only when query warning identities are absent. Synthetic warning fixtures measure missing overhead separately. No token, latency or agent-outcome claim. tools/list overhead is separate."})
        )?
    );
    Ok(())
}
