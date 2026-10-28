//! Rust item ranges always refer to the original source, never printed tokens.

use std::ops::Range;
use std::{collections::HashSet, io::Write, path::PathBuf, sync::Mutex};

use proc_macro2::Span;
use syn::spanned::Spanned;

use crate::segment::{LineIndex, TextSegment, TextSegmenter};

// Bump for pinned parser or policy changes: this versions both cache bindings
// and crash markers, so the new parser retries previously interrupted files.
pub(crate) const RUST_PREPARATION_VERSION: &str = "rust-symbols-v1";
const MAX_PARSE_BYTES: usize = 1024 * 1024;
const MAX_DELIMITER_DEPTH: usize = 128;

pub(crate) struct ParserState {
    directory: Option<PathBuf>,
    warned: Mutex<HashSet<String>>,
}

impl ParserState {
    pub(crate) fn new(directory: Option<PathBuf>) -> Self {
        Self {
            directory: directory.map(|dir| dir.join(RUST_PREPARATION_VERSION)),
            warned: Mutex::new(HashSet::new()),
        }
    }

    pub(crate) fn segment(
        &self,
        id: &crate::ResourceId,
        segmenter: &TextSegmenter,
        source: &str,
    ) -> Vec<TextSegment> {
        if !within_parse_budget(source) {
            return render(segmenter, source, vec![Unit::gap(0..source.len())]);
        }
        let Some(directory) = &self.directory else {
            return segment(segmenter, source);
        };
        let key = blake3::hash(id.as_str().as_bytes()).to_hex().to_string();
        let marker = directory.join(format!("{key}.parsing"));
        let prepared = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(directory)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
            }
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&marker)?;
            writeln!(file, "{RUST_PREPARATION_VERSION}\n{}", id.as_str())?;
            file.sync_all()
        })();
        if prepared.is_err() {
            if self
                .warned
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key)
            {
                eprintln!(
                    "warning: Rust parser recovery marker exists or is unavailable; using heading-free fallback"
                );
            }
            return render(segmenter, source, vec![Unit::gap(0..source.len())]);
        }
        let pieces = segment(segmenter, source);
        // An abort leaves this marker behind. An ordinary unwind is handled in
        // the parser thread; successfully returning evidence clears the marker.
        let _ = std::fs::remove_file(marker);
        pieces
    }
}

fn within_parse_budget(source: &str) -> bool {
    if source.len() > MAX_PARSE_BYTES {
        return false;
    }
    let mut depth = 0usize;
    let mut prefix_run = 0usize;
    let mut closure_bars = 0usize;
    for byte in source.bytes() {
        match byte {
            b'!' | b'-' | b'*' | b'&' => {
                prefix_run += 1;
                if prefix_run > MAX_DELIMITER_DEPTH {
                    return false;
                }
            }
            b'|' => {
                closure_bars += 1;
                if closure_bars > MAX_DELIMITER_DEPTH {
                    return false;
                }
            }
            b';' | b'{' | b'}' => {
                prefix_run = 0;
                closure_bars = 0;
            }
            b if b.is_ascii_whitespace() => {}
            _ => prefix_run = 0,
        }
        match byte {
            b'(' | b'[' | b'{' => {
                depth += 1;
                if depth > MAX_DELIMITER_DEPTH {
                    return false;
                }
            }
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    true
}

pub(crate) fn is_rust_file(id: &crate::ResourceId) -> bool {
    id.to_file_path().is_some_and(|path| {
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
    })
}

pub(crate) fn lexical_heading(id: &crate::ResourceId, label: &str) -> String {
    if !is_rust_file(id) {
        return label.to_owned();
    }
    label
        .split_whitespace()
        .filter(|word| {
            !matches!(
                *word,
                ">" | "fn"
                    | "impl"
                    | "mod"
                    | "trait"
                    | "extern"
                    | "struct"
                    | "enum"
                    | "union"
                    | "type"
                    | "const"
                    | "static"
            )
        })
        .map(|word| word.strip_prefix("r#").unwrap_or(word))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn segment(segmenter: &TextSegmenter, source: &str) -> Vec<TextSegment> {
    // span-locations retains source text in thread-local storage. Thread exit
    // releases it without invalidating spans owned by a library caller.
    let units = within_parse_budget(source)
        .then(|| {
            std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .name("seyk-rust-parse".into())
                    .stack_size(8 * 1024 * 1024)
                    .spawn_scoped(scope, || {
                        let parsed = Parsed::new(source).ok()?;
                        let mut units = Vec::new();
                        parsed.items(source, 0..source.len(), &parsed.file.items, &[], &mut units);
                        Some(units)
                    })
                    .ok()
                    .and_then(|thread| thread.join().ok().flatten())
            })
        })
        .flatten()
        .unwrap_or_else(|| vec![Unit::gap(0..source.len())]);
    render(segmenter, source, units)
}

fn render(segmenter: &TextSegmenter, source: &str, units: Vec<Unit>) -> Vec<TextSegment> {
    let mut merged: Vec<Unit> = Vec::new();
    for mut unit in units {
        unit.small &= unit.range.len() < segmenter.config.min_chars;
        if let Some(previous) = merged.last_mut()
            && previous.small
            && unit.small
            && previous.container == unit.container
            && unit.range.end - previous.range.start <= segmenter.config.target_chars
            && unit.range.end - previous.range.start <= segmenter.config.max_chars
        {
            previous.range.end = unit.range.end;
            previous.heading = unit.container.clone();
            continue;
        }
        merged.push(unit);
    }
    let lines = LineIndex::new(source);
    let mut out = Vec::new();
    for unit in merged {
        let mut start = unit.range.start;
        while start < unit.range.end {
            let limit = start
                .saturating_add(segmenter.config.max_chars)
                .min(unit.range.end);
            let mut end = limit;
            while !source.is_char_boundary(end) {
                end -= 1;
            }
            if end == start {
                end += source[start..].chars().next().unwrap().len_utf8();
            } else if end < unit.range.end {
                let mut target = segmenter
                    .config
                    .target_chars
                    .min(end - start)
                    .max((end - start) / 2);
                while !source.is_char_boundary(start + target) {
                    target -= 1;
                }
                if let Some(newline) = source[start + target..end].rfind('\n') {
                    end = start + target + newline + 1;
                }
            }
            let text = &source[start..end];
            if !text.trim().is_empty() {
                out.push(TextSegment {
                    byte_range: start..end,
                    line_start: lines.line_of(start),
                    line_end: lines.line_of(end - 1),
                    heading: unit.heading.clone(),
                    text: text.to_owned(),
                });
            }
            start = end;
        }
    }
    out
}

struct Unit {
    range: Range<usize>,
    heading: Option<String>,
    container: Option<String>,
    small: bool,
}

impl Unit {
    fn gap(range: Range<usize>) -> Self {
        Self {
            range,
            heading: None,
            container: None,
            small: false,
        }
    }
}

fn heading(context: &[String]) -> Option<String> {
    let mut start = 0;
    while start < context.len() {
        let label = context[start..].join(" > ");
        if label.len() <= 256 {
            return Some(label);
        }
        start += 1;
    }
    None
}

fn named(context: &[String], kind: &str, ident: &syn::Ident) -> Vec<String> {
    let mut context = context.to_vec();
    context.push(format!("{kind} {ident}"));
    context
}

fn type_name(ty: &syn::Type) -> Option<&syn::Ident> {
    match ty {
        syn::Type::Path(path) => path.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

struct Parsed {
    file: syn::File,
    offset: usize,
}

impl Parsed {
    fn new(source: &str) -> syn::Result<Self> {
        let file = syn::parse_file(source)?;
        // parse_file strips these prefixes before creating the token stream.
        let offset = usize::from(source.starts_with('\u{feff}')) * '\u{feff}'.len_utf8()
            + file.shebang.as_ref().map_or(0, String::len);
        Ok(Self { file, offset })
    }

    fn range(&self, span: Span) -> Range<usize> {
        let range = span.byte_range();
        range.start + self.offset..range.end + self.offset
    }

    fn gap(&self, source: &str, range: Range<usize>, out: &mut Vec<Unit>) {
        if !source[range.clone()].trim().is_empty() {
            out.push(Unit::gap(range));
        }
    }

    fn leaf(&self, range: Range<usize>, context: &[String], label: &[String], out: &mut Vec<Unit>) {
        out.push(Unit {
            range,
            heading: heading(label),
            container: heading(context),
            small: false,
        });
    }

    fn items(
        &self,
        source: &str,
        bounds: Range<usize>,
        items: &[syn::Item],
        context: &[String],
        out: &mut Vec<Unit>,
    ) {
        let mut cursor = bounds.start;
        for item in items {
            let range = self.range(item.span());
            assert!(range.start >= cursor && range.end <= bounds.end);
            self.gap(source, cursor..range.start, out);
            self.item(source, item, context, out);
            cursor = range.end;
        }
        self.gap(source, cursor..bounds.end, out);
    }

    fn item(&self, source: &str, item: &syn::Item, context: &[String], out: &mut Vec<Unit>) {
        let range = self.range(item.span());
        let leaf = match item {
            syn::Item::Fn(i) => Some(named(context, "fn", &i.sig.ident)),
            syn::Item::Struct(i) => Some(named(context, "struct", &i.ident)),
            syn::Item::Enum(i) => Some(named(context, "enum", &i.ident)),
            syn::Item::Union(i) => Some(named(context, "union", &i.ident)),
            syn::Item::Const(i) => Some(named(context, "const", &i.ident)),
            syn::Item::Static(i) => Some(named(context, "static", &i.ident)),
            syn::Item::Type(i) => Some(named(context, "type", &i.ident)),
            syn::Item::TraitAlias(i) => Some(named(context, "trait", &i.ident)),
            syn::Item::Mod(i) => {
                let label = named(context, "mod", &i.ident);
                if let Some((brace, items)) = &i.content {
                    let open = self.range(brace.span.open()).end;
                    let close = self.range(brace.span.close()).start;
                    self.leaf(range.start..open, context, &label, out);
                    self.items(source, open..close, items, &label, out);
                    self.leaf(close..range.end, context, &label, out);
                    return;
                }
                Some(label)
            }
            syn::Item::Impl(i) => {
                let mut label = context.to_vec();
                if let Some(name) = type_name(&i.self_ty) {
                    label.push(format!("impl {name}"));
                }
                self.children(
                    source,
                    range,
                    &i.brace_token,
                    &i.items,
                    context,
                    &label,
                    out,
                    |item| match item {
                        syn::ImplItem::Fn(i) => Some(named(&label, "fn", &i.sig.ident)),
                        syn::ImplItem::Const(i) => Some(named(&label, "const", &i.ident)),
                        syn::ImplItem::Type(i) => Some(named(&label, "type", &i.ident)),
                        _ => None,
                    },
                );
                return;
            }
            syn::Item::Trait(i) => {
                let label = named(context, "trait", &i.ident);
                self.children(
                    source,
                    range,
                    &i.brace_token,
                    &i.items,
                    context,
                    &label,
                    out,
                    |item| match item {
                        syn::TraitItem::Fn(i) => Some(named(&label, "fn", &i.sig.ident)),
                        syn::TraitItem::Const(i) => Some(named(&label, "const", &i.ident)),
                        syn::TraitItem::Type(i) => Some(named(&label, "type", &i.ident)),
                        _ => None,
                    },
                );
                return;
            }
            syn::Item::ForeignMod(i) => {
                let mut label = context.to_vec();
                label.push("extern".to_owned());
                self.children(
                    source,
                    range,
                    &i.brace_token,
                    &i.items,
                    context,
                    &label,
                    out,
                    |item| match item {
                        syn::ForeignItem::Fn(i) => Some(named(&label, "fn", &i.sig.ident)),
                        syn::ForeignItem::Static(i) => Some(named(&label, "static", &i.ident)),
                        syn::ForeignItem::Type(i) => Some(named(&label, "type", &i.ident)),
                        _ => None,
                    },
                );
                return;
            }
            // Macro token trees and future syntax remain opaque, with no guessed label.
            _ => None,
        };
        let mergeable =
            leaf.is_some() || matches!(item, syn::Item::Use(_) | syn::Item::ExternCrate(_));
        self.leaf(range, context, leaf.as_deref().unwrap_or(&[]), out);
        out.last_mut().unwrap().small = mergeable;
    }

    #[allow(clippy::too_many_arguments)]
    fn children<T: Spanned>(
        &self,
        source: &str,
        range: Range<usize>,
        brace: &syn::token::Brace,
        items: &[T],
        parent: &[String],
        context: &[String],
        out: &mut Vec<Unit>,
        label: impl Fn(&T) -> Option<Vec<String>>,
    ) {
        let open = self.range(brace.span.open()).end;
        let close = self.range(brace.span.close()).start;
        self.leaf(range.start..open, parent, context, out);
        let mut cursor = open;
        for item in items {
            let range = self.range(item.span());
            assert!(range.start >= cursor && range.end <= close);
            self.gap(source, cursor..range.start, out);
            cursor = range.end;
            let label = label(item);
            self.leaf(range, context, label.as_deref().unwrap_or(&[]), out);
            out.last_mut().unwrap().small = label.is_some();
        }
        self.gap(source, cursor..close, out);
        self.leaf(close..range.end, parent, context, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{TextFormat, TextSegmenterConfig};

    fn check(source: &str, segmenter: &TextSegmenter) -> Vec<TextSegment> {
        let segments = segment(segmenter, source);
        let mut covered = vec![false; source.len()];
        let mut previous_end = 0;
        for piece in &segments {
            assert!(piece.byte_range.start >= previous_end);
            assert_eq!(piece.text, source[piece.byte_range.clone()]);
            assert!(
                piece.text.len() <= segmenter.config.max_chars || piece.text.chars().count() == 1
            );
            assert_eq!(
                piece.line_start,
                source[..piece.byte_range.start]
                    .bytes()
                    .filter(|&b| b == b'\n')
                    .count() as u64
                    + 1
            );
            assert_eq!(
                piece.line_end,
                source.as_bytes()[..piece.byte_range.end - 1]
                    .iter()
                    .copied()
                    .filter(|&b| b == b'\n')
                    .count() as u64
                    + 1
            );
            covered[piece.byte_range.clone()].fill(true);
            previous_end = piece.byte_range.end;
        }
        for (offset, c) in source.char_indices() {
            assert!(
                c.is_whitespace() || covered[offset..offset + c.len_utf8()].iter().all(|b| *b),
                "uncovered {c:?} at {offset}"
            );
        }
        segments
    }

    fn no_merge() -> TextSegmenter {
        TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            ..Default::default()
        })
    }

    #[test]
    fn segmentation_does_not_retain_sources_or_invalidate_spans_on_caller_thread() {
        fn caller_spans(parse: bool) -> (String, String) {
            std::thread::spawn(move || {
                let before: proc_macro2::TokenStream = "caller_before".parse().unwrap();
                let before = before.into_iter().next().unwrap().span();
                let original = format!("{before:?}");
                if parse {
                    segment(&TextSegmenter::default(), "fn isolated() {}");
                }
                assert_eq!(before.source_text().as_deref(), Some("caller_before"));
                assert_eq!(format!("{before:?}"), original);
                let after: proc_macro2::TokenStream = "caller_after".parse().unwrap();
                let after = after.into_iter().next().unwrap().span();
                (original, format!("{after:?}"))
            })
            .join()
            .unwrap()
        }
        // Debug includes global source-map positions, unlike byte_range(),
        // which is relative to each source. Any caller-thread parse changes it.
        assert_eq!(caller_spans(true), caller_spans(false));
    }

    #[test]
    fn interrupted_parser_markers_survive_restart_and_force_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let id = crate::ResourceId::from_path(temp.path().join("file.rs")).unwrap();
        let key = blake3::hash(id.as_str().as_bytes()).to_hex().to_string();
        let dir = temp.path().join(RUST_PREPARATION_VERSION);
        std::fs::create_dir(&dir).unwrap();
        let marker = dir.join(format!("{key}.parsing"));
        std::fs::write(&marker, format!("{RUST_PREPARATION_VERSION}\n{id}")).unwrap();
        for _ in 0..2 {
            let state = ParserState::new(Some(temp.path().to_owned()));
            let pieces = state.segment(&id, &no_merge(), "fn recovered() {}");
            assert!(pieces.iter().all(|p| p.heading.is_none()));
            assert!(marker.exists());
        }
        std::fs::remove_file(&marker).unwrap();
        let state = ParserState::new(Some(temp.path().to_owned()));
        let pieces = state.segment(&id, &no_merge(), "fn recovered() {}");
        assert_eq!(pieces[0].heading.as_deref(), Some("fn recovered"));
        assert!(!marker.exists());
    }

    #[test]
    fn parse_budgets_fall_back_before_recursive_parsing() {
        for source in [
            format!("fn large() {{ /*{}*/ }}", "x".repeat(MAX_PARSE_BYTES)),
            format!("fn deep() {{ {}0{}; }}", "(".repeat(129), ")".repeat(129)),
            format!("fn prefix() {{ let _ = {}true; }}", "! ".repeat(129)),
            format!("fn closure() {{ let _ = {}0; }}", "|| ".repeat(65)),
        ] {
            assert!(!within_parse_budget(&source));
            assert!(
                check(&source, &no_merge())
                    .iter()
                    .all(|p| p.heading.is_none())
            );
        }
    }

    #[test]
    fn headings_drop_outer_context_before_a_long_leaf() {
        let source = format!("mod {} {{ fn leaf() {{}} }}", "a".repeat(250));
        let pieces = check(&source, &no_merge());
        assert!(
            pieces
                .iter()
                .any(|p| p.heading.as_deref() == Some("fn leaf"))
        );
        assert!(
            pieces
                .iter()
                .all(|p| p.heading.as_ref().is_none_or(|h| h.len() <= 256))
        );
        let source = format!("fn {}() {{}}", "é".repeat(140));
        assert!(
            check(&source, &no_merge())
                .iter()
                .all(|p| p.heading.is_none())
        );
    }

    #[test]
    fn lexical_labels_keep_only_identifiers_and_preserve_other_resources() {
        let temp = tempfile::tempdir().unwrap();
        let id = crate::ResourceId::from_path(temp.path().join("code.rs")).unwrap();
        let label = "mod café > impl Foo > fn r#type";
        assert_eq!(lexical_heading(&id, label), "café Foo type");
        let id = "memory://corpus/code.rs".parse().unwrap();
        assert_eq!(lexical_heading(&id, label), label);
    }

    #[test]
    fn containers_are_always_decomposed_and_macros_stay_opaque() {
        let source = r####"#![allow(dead_code)]
//! inner docs
mod café {
    impl<const N: usize> Foo<{ N }> {
        #[cfg(test)]
        /// Docs for a getter.
        pub(crate) async unsafe fn r#type<'a>(&'a self) where Self: Sized {
            /* outer /* inner } */ */
            let _ = (r###" } fn fake() {} "###, br#"{"#, c"}", cr#"{"#);
            let _ = ('{', '"', '\'', b'}');
            'outer: loop { break 'outer; }
            let _: fn(u8) -> u8 = |x| x;
        }
        const fn other() {}
    }
    trait Tiny { fn a(); fn b() {} }
    unsafe extern "C" { fn ext(); fn ext2(); }
    macro_rules! opaque { () => { fn fake() {} } }
    path::opaque!({ fn fake() {} });
}
"####;
        let pieces = check(source, &no_merge());
        let labels: Vec<_> = pieces.iter().filter_map(|p| p.heading.as_deref()).collect();
        assert!(labels.contains(&"mod café > impl Foo > fn r#type"));
        assert!(labels.contains(&"mod café > impl Foo > fn other"));
        assert!(labels.contains(&"mod café > trait Tiny > fn a"));
        assert!(labels.contains(&"mod café > trait Tiny > fn b"));
        assert!(labels.contains(&"mod café > extern > fn ext"));
        assert!(!labels.iter().any(|h| h.contains("fake")));
        let method = pieces
            .iter()
            .find(|p| p.heading.as_deref() == Some("mod café > impl Foo > fn r#type"))
            .unwrap();
        assert!(method.text.starts_with("#[cfg(test)]"));
        assert!(method.text.contains("Docs for a getter"));
    }

    #[test]
    fn tiny_siblings_merge_only_within_their_container() {
        let source = "impl Foo { fn a() {} fn b() {} } impl Bar { fn c() {} fn d() {} }";
        let pieces = check(source, &TextSegmenter::default());
        let merged = pieces
            .iter()
            .find(|p| p.text.contains("fn a") && p.text.contains("fn b"))
            .unwrap();
        assert_eq!(merged.heading.as_deref(), Some("impl Foo"));
        assert!(
            !pieces
                .iter()
                .any(|p| p.text.contains("fn b") && p.text.contains("fn c"))
        );
        assert!(
            !check(source, &no_merge())
                .iter()
                .any(|p| p.text.contains("fn a") && p.text.contains("fn b"))
        );
    }

    #[test]
    fn opaque_items_never_inherit_a_sibling_heading() {
        let pieces = check(
            "impl Foo { fn a() {} opaque!(); fn b() {} }",
            &TextSegmenter::default(),
        );
        let opaque = pieces.iter().find(|p| p.text.contains("opaque!")).unwrap();
        assert_eq!(opaque.heading, None);
        assert_eq!(opaque.text, "opaque!();");
    }

    #[test]
    fn long_functions_keep_their_label_and_stay_inside_the_item() {
        let source = format!(
            "impl Foo {{ fn café() {{\n{}\n}} fn after() {{}} }}",
            "let _ = \"🙂\";\n".repeat(400)
        );
        let pieces = check(&source, &no_merge());
        let parsed = Parsed::new(&source).unwrap();
        let syn::Item::Impl(i) = &parsed.file.items[0] else {
            panic!()
        };
        let range = parsed.range(i.items[0].span());
        let function: Vec<_> = pieces
            .iter()
            .filter(|p| p.heading.as_deref() == Some("impl Foo > fn café"))
            .collect();
        assert!(function.len() > 1);
        assert!(function[0].text.contains("fn café"));
        for piece in function {
            assert!(range.start <= piece.byte_range.start && piece.byte_range.end <= range.end);
        }
    }

    #[test]
    fn invalid_source_and_long_lines_use_bounded_heading_free_fallback() {
        for source in [
            "fn incomplete( {".to_owned(),
            format!(
                "unknown {} syntax\n\n{}tail",
                "🙂".repeat(2000),
                " ".repeat(5000)
            ),
        ] {
            let pieces = check(&source, &TextSegmenter::default());
            assert!(pieces.iter().all(|p| p.heading.is_none()));
        }
        let segmenter = TextSegmenter::new(TextSegmenterConfig {
            max_chars: 1,
            target_chars: 0,
            ..Default::default()
        });
        check("🙂 incomplete", &segmenter);
    }

    #[test]
    fn repository_rust_corpus_keeps_exact_complete_bounded_slices() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let segmenter = TextSegmenter::default();
        let mut rust_count = 0;
        let mut legacy_count = 0;
        let mut files = 0;
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !matches!(e.file_name().to_str(), Some("target" | ".git" | ".anvil")))
        {
            let entry = entry.unwrap();
            if !entry.file_type().is_file()
                || entry.path().extension().is_none_or(|ext| ext != "rs")
            {
                continue;
            }
            let source = std::fs::read_to_string(entry.path()).unwrap();
            let pieces = check(&source, &segmenter);
            if let Ok(parsed) = Parsed::new(&source) {
                let mut units = Vec::new();
                parsed.items(
                    &source,
                    0..source.len(),
                    &parsed.file.items,
                    &[],
                    &mut units,
                );
                for piece in &pieces {
                    let Some(label) = &piece.heading else {
                        continue;
                    };
                    if label.rsplit(" > ").next().unwrap().starts_with("fn ") {
                        assert!(
                            units.iter().any(|u| u.heading.as_ref() == Some(label)
                                && u.range.start <= piece.byte_range.start
                                && piece.byte_range.end <= u.range.end),
                            "{}: {label}",
                            entry.path().display()
                        );
                    }
                }
                for unit in units.iter().filter(|u| {
                    u.heading
                        .as_deref()
                        .is_some_and(|h| h.rsplit(" > ").next().unwrap().starts_with("fn "))
                }) {
                    if let Some(first) = pieces.iter().find(|p| {
                        p.heading == unit.heading
                            && p.byte_range.start >= unit.range.start
                            && p.byte_range.end <= unit.range.end
                    }) {
                        let leaf = unit.heading.as_ref().unwrap().rsplit(" > ").next().unwrap();
                        assert!(
                            first.text.contains(leaf),
                            "{}: missing declaration {leaf}",
                            entry.path().display()
                        );
                    }
                }
            } else {
                assert!(pieces.iter().all(|p| p.heading.is_none()));
            }
            rust_count += pieces.len();
            legacy_count += segmenter.segment(&source, TextFormat::Code).len();
            files += 1;
        }
        assert!(files > 100);
        eprintln!(
            "RUST_CORPUS files={files} symbol_segments={rust_count} legacy_segments={legacy_count}"
        );
    }

    #[test]
    fn parser_spans_preserve_original_bytes() {
        for prefix in [
            "",
            "\u{feff}",
            "#!/usr/bin/env rustx\n",
            "\u{feff}#!/é/rustx\r\n",
        ] {
            for newline in ["\n", "\r\n"] {
                let item = format!("pub fn café() {{{newline}    let s = \"été\";{newline}}}");
                let source = format!("{prefix}// café{newline}{item}{newline}");
                let parsed = Parsed::new(&source).unwrap();
                assert_eq!(parsed.file.items.len(), 1);
                let range = parsed.range(parsed.file.items[0].span());
                assert_eq!(&source[range.clone()], item, "{source:?}: {range:?}");
                let syn::Item::Fn(function) = &parsed.file.items[0] else {
                    panic!()
                };
                assert_eq!(&source[parsed.range(function.sig.ident.span())], "café");
                assert_eq!(
                    &source[parsed.range(function.block.brace_token.span.open())],
                    "{"
                );
                assert_eq!(
                    &source[parsed.range(function.block.brace_token.span.close())],
                    "}"
                );
            }
        }
    }
}
