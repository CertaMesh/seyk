//! Lexical (keyword) matching, the half of hybrid search that embeddings miss:
//! exact names, identifiers and rare words.
//!
//! Text is tokenized the way SQLite FTS5's `unicode61 remove_diacritics 2`
//! tokenizer does it: letters and digits form tokens, everything else
//! separates them, case and accents are ignored. `SqliteCache` stores these
//! tokens in a contentless FTS5 table; `MemoryCache` keeps them in a small BM25
//! index built here. Both rank the same fields with the same weights.

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

/// The fields a segment is matched on, with their BM25 weights. A query word in
/// the file name counts ten times as much as one in the body text.
pub const FIELDS: [(&str, f64); 4] = [
    ("name", 10.0),
    ("path", 3.0),
    ("heading", 4.0),
    ("body", 1.0),
];

/// Function words that would match nearly every segment and only add noise to a
/// natural-language question. Identifiers and names never consist of these.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so",
    "some", "that", "the", "their", "them", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
];

/// Lowercased, accent-free tokens of `text`.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.nfd() {
        if c.is_alphanumeric() {
            cur.extend(c.to_lowercase());
        } else if is_combining_mark(c) {
            // Accent of the previous letter: dropped, so "café" matches "cafe".
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn is_combining_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

/// What a query asks for lexically: groups of tokens that must appear next to
/// each other. `Session::restore()` is one group, `session restore`; plain
/// words are groups of one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LexicalQuery {
    pub groups: Vec<Vec<String>>,
}

impl LexicalQuery {
    /// Build from one or more phrasings. Whitespace separates groups; within a
    /// group, punctuation joins tokens into a phrase. Single-token stopwords
    /// are dropped, so a question matches on its content words.
    pub fn new<'a>(queries: impl IntoIterator<Item = &'a str>) -> Self {
        let mut groups: Vec<Vec<String>> = Vec::new();
        for q in queries {
            for word in q.split_whitespace() {
                let tokens = tokenize(word);
                let keep = match tokens.as_slice() {
                    [] => false,
                    [t] => !STOPWORDS.contains(&t.as_str()),
                    _ => true,
                };
                if keep && !groups.contains(&tokens) {
                    groups.push(tokens);
                }
            }
        }
        Self { groups }
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// An FTS5 `MATCH` expression: every group quoted (so no user text is ever
    /// parsed as FTS syntax) and OR-ed, ranked by BM25.
    pub fn fts5(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        Some(
            self.groups
                .iter()
                .map(|g| format!("\"{}\"", g.join(" ")))
                .collect::<Vec<_>>()
                .join(" OR "),
        )
    }
}

/// One indexed segment's fields, already tokenized, in [`FIELDS`] order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LexicalDoc {
    pub fields: [Vec<String>; 4],
}

impl LexicalDoc {
    pub fn new(name: &str, path: &str, heading: Option<&str>, body: &str) -> Self {
        Self {
            fields: [
                tokenize(name),
                tokenize(path),
                heading.map(tokenize).unwrap_or_default(),
                tokenize(body),
            ],
        }
    }

    /// Whether `group` occurs as consecutive tokens in field `f`.
    fn phrase_count(&self, f: usize, group: &[String]) -> usize {
        let tokens = &self.fields[f];
        if group.is_empty() || tokens.len() < group.len() {
            return 0;
        }
        tokens.windows(group.len()).filter(|w| *w == group).count()
    }
}

/// BM25 over a set of documents, used by the in-memory cache. Matches FTS5's
/// `bm25()` in shape (k1 = 1.2, b = 0.75, field weights) so both caches rank
/// alike; absolute scores differ, and fusion only uses ranks.
pub fn bm25<'d, K: Clone>(
    docs: impl IntoIterator<Item = (K, &'d LexicalDoc)>,
    query: &LexicalQuery,
) -> Vec<(K, f64)> {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    let docs: Vec<(K, &LexicalDoc)> = docs.into_iter().collect();
    let n = docs.len() as f64;
    if n == 0.0 || query.is_empty() {
        return Vec::new();
    }
    let avg_len: Vec<f64> = (0..4)
        .map(|f| docs.iter().map(|(_, d)| d.fields[f].len()).sum::<usize>() as f64 / n)
        .collect();
    let doc_freq: HashMap<&Vec<String>, f64> = query
        .groups
        .iter()
        .map(|g| {
            let df = docs
                .iter()
                .filter(|(_, d)| (0..4).any(|f| d.phrase_count(f, g) > 0))
                .count();
            (g, df as f64)
        })
        .collect();
    let mut scored: Vec<(K, f64)> = docs
        .iter()
        .filter_map(|(k, d)| {
            let mut score = 0.0;
            for g in &query.groups {
                let df = doc_freq[g];
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                for (f, (_, weight)) in FIELDS.iter().enumerate() {
                    let tf = d.phrase_count(f, g) as f64;
                    if tf == 0.0 {
                        continue;
                    }
                    let len_norm = if avg_len[f] > 0.0 {
                        1.0 - B + B * d.fields[f].len() as f64 / avg_len[f]
                    } else {
                        1.0
                    };
                    score += weight * idf * tf * (K1 + 1.0) / (tf + K1 * len_norm);
                }
            }
            (score > 0.0).then(|| (k.clone(), score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_ignore_case_accents_and_punctuation() {
        assert_eq!(
            tokenize("Café-Zürich, INC-18372!"),
            ["cafe", "zurich", "inc", "18372"]
        );
        assert_eq!(tokenize("Session::restore()"), ["session", "restore"]);
        assert_eq!(tokenize("invoice_38192"), ["invoice", "38192"]);
    }

    #[test]
    fn queries_keep_identifiers_as_phrases_and_drop_stopwords() {
        let q = LexicalQuery::new(["where is INC-18372 and the father.md"]);
        assert_eq!(
            q.groups,
            [vec!["inc", "18372"], vec!["father", "md"]]
                .map(|g| g.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        );
        assert_eq!(q.fts5().unwrap(), "\"inc 18372\" OR \"father md\"");
        assert!(LexicalQuery::new(["where is the"]).fts5().is_none());
    }

    #[test]
    fn quotes_in_queries_cannot_break_out_of_a_phrase() {
        // Tokens never contain quotes or FTS5 operators.
        let q = LexicalQuery::new([r#"a" OR body:x NEAR(y)"#]);
        assert!(q.fts5().unwrap().split('"').all(|p| !p.contains(':')));
    }

    #[test]
    fn filename_matches_outrank_body_mentions() {
        let named = LexicalDoc::new("father.md", "book/father.md", None, "Notes on him.");
        let mentioned = LexicalDoc::new(
            "ideas.md",
            "book/ideas.md",
            None,
            "My father and the md file.",
        );
        let ranked = bm25(
            [("named", &named), ("mentioned", &mentioned)],
            &LexicalQuery::new(["father.md"]),
        );
        assert_eq!(ranked[0].0, "named");
    }
}
