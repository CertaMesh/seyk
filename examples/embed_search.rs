//! Embed Seyk in a Rust program: search a folder, then extract the top hit as
//! evidence.
//!
//! This uses `SearchEngine::lexical()`, which ranks by keywords and file names
//! and needs no model weights, so it runs anywhere. For hybrid search with
//! meaning, build the engine with an embedder instead (see docs/library.md).
//!
//! ```sh
//! cargo run --example embed_search -- "how do background jobs write to the database?"
//! cargo run --example embed_search -- "session cookie" path/to/folder
//! ```

use std::path::{Path, PathBuf};

use seyk::{ResourceLocator, SearchEngine};

fn main() -> seyk::Result<()> {
    let mut args = std::env::args().skip(1);
    let query = args
        .next()
        .unwrap_or_else(|| "how do background jobs write to the database?".to_owned());
    let folder = args.next().map(PathBuf::from).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/agent-decisions/decisions")
    });
    // Canonical, so hit paths can be shown relative to it.
    let folder = folder.canonicalize().map_err(|source| seyk::Error::Io {
        path: folder.clone(),
        source,
    })?;

    let engine = SearchEngine::lexical()?;
    let results = engine
        .search(query.as_str())
        .paths([&folder])
        .limit(3)
        .execute()?;

    println!("query: {query}");
    for hit in &results.hits {
        println!(
            "{:.3}  {}{}  {}",
            hit.score,
            display_path(&folder, hit),
            lines(&hit.locator),
            hit.heading.as_deref().unwrap_or(""),
        );
    }

    let Some(top) = results.hits.first() else {
        println!("no hits");
        return Ok(());
    };
    // Extraction re-reads the file and checks it against the hash recorded at
    // indexing. A changed file is an error, never a different passage.
    let evidence = engine.extract(top)?;
    println!(
        "\nextracted {}{} (file BLAKE3 {}…):",
        display_path(&folder, top),
        lines(&evidence.locator),
        &evidence.metadata.content_hash[..12],
    );
    println!("{}", evidence.text.trim_end());
    Ok(())
}

fn display_path(folder: &Path, hit: &seyk::SearchHit) -> String {
    hit.resource
        .to_file_path()
        .and_then(|p| p.strip_prefix(folder).ok().map(Path::to_path_buf))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| hit.resource.to_string())
}

fn lines(locator: &ResourceLocator) -> String {
    match locator {
        ResourceLocator::Text {
            line_start: Some(start),
            line_end: Some(end),
            ..
        } => format!(":{start}-{end}"),
        _ => String::new(),
    }
}
