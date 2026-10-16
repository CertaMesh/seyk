//! Lexical (keyword) matching, the half of hybrid search that embeddings miss:
//! exact names, identifiers and rare words.
//!
//! Both caches use SQLite FTS5's `unicode61 remove_diacritics 2` tokenizer.
//! Persistent fields contain sorted tokens, including duplicates: frequencies
//! and field lengths survive, but positions reveal no source word order.
//! Identifier groups require all their tokens in the same field, in any order.

use std::cell::RefCell;

use rusqlite::Connection;

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

thread_local! {
    // Reuse SQLite's own tokenizer instead of approximating its Unicode 6.1
    // character classes and case folding with Rust's newer Unicode tables.
    // This scratch index is memory-only, including for MemoryCache users.
    static TOKENIZER: RefCell<Connection> = RefCell::new({
        let conn = Connection::open_in_memory().expect("open in-memory tokenizer");
        conn.execute_batch(
            "PRAGMA temp_store = MEMORY;
             CREATE VIRTUAL TABLE tokens USING fts5(text, content='', contentless_delete=1,
                 tokenize='unicode61 remove_diacritics 2');
             CREATE VIRTUAL TABLE vocabulary USING fts5vocab(tokens, instance);"
        ).expect("initialize bundled FTS5 tokenizer");
        conn
    });
}

/// Tokens normalized by the same tokenizer that indexes persistent fields.
pub fn tokenize(text: &str) -> Vec<String> {
    TOKENIZER.with(|cell| {
        let mut conn = cell.borrow_mut();
        let tx = conn.transaction().expect("begin tokenization");
        tx.execute("INSERT INTO tokens(rowid, text) VALUES (1, ?1)", [text])
            .expect("tokenize text");
        let tokens = tx
            .prepare_cached("SELECT term FROM vocabulary ORDER BY offset")
            .expect("prepare token scan")
            .query_map([], |row| row.get(0))
            .expect("scan tokens")
            .collect::<rusqlite::Result<Vec<String>>>()
            .expect("read tokens");
        // Roll back the scratch row and its positions even if reading panics.
        tx.rollback().expect("discard tokenizer scratch row");
        tokens
    })
}

/// FTS input preserving term frequencies but no original token positions.
pub fn order_free_text(text: &str) -> String {
    let mut tokens = tokenize(text);
    tokens.sort_unstable();
    tokens.join(" ")
}

/// OR-ed groups of tokens. All tokens of a group must occur in one field,
/// without requiring adjacency or source order; plain words are groups of one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LexicalQuery {
    pub groups: Vec<Vec<String>>,
}

impl LexicalQuery {
    /// Build from one or more phrasings. Whitespace separates groups; within a
    /// group, punctuation joins tokens into a conjunction. Single-token stopwords
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

    /// An FTS5 `MATCH` expression: quoted tokens AND-ed within each field,
    /// groups OR-ed. Only our fixed field names are parsed as FTS syntax.
    pub fn fts5(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        Some(
            self.groups
                .iter()
                .map(|g| {
                    let terms = g
                        .iter()
                        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
                        .collect::<Vec<_>>()
                        .join(" AND ");
                    if g.len() == 1 {
                        terms
                    } else {
                        // A bare AND could satisfy an identifier across unrelated fields.
                        format!(
                            "({})",
                            FIELDS
                                .iter()
                                .map(|(field, _)| format!("{field}:({terms})"))
                                .collect::<Vec<_>>()
                                .join(" OR ")
                        )
                    }
                })
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

    /// Return only query tokens whose complete group matches a single field.
    pub fn matched_terms(&self, query: &LexicalQuery) -> Vec<String> {
        let mut terms = Vec::new();
        for group in &query.groups {
            if (0..FIELDS.len()).any(|field| self.matches_group(field, group)) {
                for term in group {
                    if !terms.contains(term) {
                        terms.push(term.clone());
                    }
                }
            }
        }
        terms
    }

    fn term_count(&self, field: usize, term: &str) -> usize {
        self.fields[field].iter().filter(|t| *t == term).count()
    }

    fn matches_group(&self, field: usize, group: &[String]) -> bool {
        !group.is_empty() && group.iter().all(|t| self.term_count(field, t) > 0)
    }

    fn len(&self) -> usize {
        self.fields.iter().map(Vec::len).sum()
    }
}

/// BM25 with FTS5's k1, b, IDF floor, weighted term frequencies and total
/// document-length normalization. Each quoted token is an FTS5 phrase; a
/// multi-token group's field-filtered phrases contribute only in matching fields.
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
    let avg_len = docs.iter().map(|(_, d)| d.len()).sum::<usize>() as f64 / n;
    // (group, token, optional column restriction, inverse document frequency)
    let mut phrases = Vec::new();
    for group in &query.groups {
        let fields: Vec<Option<usize>> = if group.len() == 1 {
            vec![None]
        } else {
            (0..FIELDS.len()).map(Some).collect()
        };
        for field in fields {
            for term in group {
                let df = docs
                    .iter()
                    .filter(|(_, d)| match field {
                        Some(f) => d.term_count(f, term) > 0,
                        None => (0..FIELDS.len()).any(|f| d.term_count(f, term) > 0),
                    })
                    .count() as f64;
                let idf = ((n - df + 0.5) / (df + 0.5)).ln().max(1e-6);
                phrases.push((group, term, field, idf));
            }
        }
    }
    let mut scored: Vec<(K, f64)> = docs
        .iter()
        .filter_map(|(key, doc)| {
            let mut score = 0.0;
            for (group, term, field, idf) in &phrases {
                let tf = match field {
                    Some(f) if doc.matches_group(*f, group) => {
                        FIELDS[*f].1 * doc.term_count(*f, term) as f64
                    }
                    Some(_) => 0.0,
                    None => FIELDS
                        .iter()
                        .enumerate()
                        .map(|(f, (_, w))| w * doc.term_count(f, term) as f64)
                        .sum(),
                };
                if tf > 0.0 {
                    let norm = 1.0 - B + B * doc.len() as f64 / avg_len;
                    score += idf * tf * (K1 + 1.0) / (tf + K1 * norm);
                }
            }
            (score > 0.0).then(|| (key.clone(), score))
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
    fn queries_keep_identifiers_as_conjunctions_and_drop_stopwords() {
        let q = LexicalQuery::new(["where is INC-18372 and the father.md"]);
        assert_eq!(
            q.groups,
            [vec!["inc", "18372"], vec!["father", "md"]]
                .map(|g| g.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        );
        assert_eq!(
            q.fts5().unwrap(),
            concat!(
                "(name:(\"inc\" AND \"18372\") OR path:(\"inc\" AND \"18372\") OR ",
                "heading:(\"inc\" AND \"18372\") OR body:(\"inc\" AND \"18372\")) OR ",
                "(name:(\"father\" AND \"md\") OR path:(\"father\" AND \"md\") OR ",
                "heading:(\"father\" AND \"md\") OR body:(\"father\" AND \"md\"))"
            )
        );
        assert!(LexicalQuery::new(["where is the"]).fts5().is_none());
    }

    #[test]
    fn quotes_in_queries_cannot_break_out_of_a_phrase() {
        // Tokens never contain quotes or FTS5 operators.
        let q = LexicalQuery::new([r#"a" OR body:x NEAR(y)"#]);
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE f USING fts5(name,path,heading,body);
            INSERT INTO f VALUES('','','','unrelated');",
        )
        .unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM f WHERE f MATCH ?1",
                [q.fts5().unwrap()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn sorted_fields_round_trip_through_unicode61() {
        // Greek final sigma, Indic combining marks, Roman numerals and accents
        // expose differences from Rust is_alphanumeric() + lowercase + NFD.
        let text = "Zürich Café κόσμος नमस्ते 東京 Привет Straße ẞ aⅣ2 v2alpha INC-18372 Café";
        let sorted = order_free_text(text);
        let mut expected = tokenize(text);
        expected.sort_unstable();
        assert_eq!(tokenize(&sorted), expected);
        assert_eq!(order_free_text(&sorted), sorted);
        assert_eq!(
            order_free_text("zeta alpha ZETA beta"),
            "alpha beta zeta zeta"
        );
        assert_eq!(tokenize("κόσμος नमस्ते aⅣ2"), ["κόσμοσ", "नमस", "त", "aⅳ2"]);
    }

    #[test]
    fn memory_scores_match_fts5_conjunctions_and_field_weights() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE docs USING fts5(name,path,heading,body,
            content='',contentless_delete=1,tokenize='unicode61 remove_diacritics 2');",
        )
        .unwrap();
        let fields = [
            ["session", "", "", "restore"], // Must not join across fields.
            ["session restore", "", "", "session"],
            ["", "", "", "restore extra session session"],
            ["", "session restore", "", "different"],
            ["", "", "restore session", "father"],
            ["father", "", "", "unrelated unrelated"],
            ["other", "", "", "other other"],
        ];
        let mut docs = Vec::new();
        for (i, f) in fields.iter().enumerate() {
            docs.push(LexicalDoc::new(f[0], f[1], Some(f[2]), f[3]));
            conn.execute(
                "INSERT INTO docs(rowid,name,path,heading,body) VALUES(?1,?2,?3,?4,?5)",
                rusqlite::params![
                    i as i64,
                    order_free_text(f[0]),
                    order_free_text(f[1]),
                    order_free_text(f[2]),
                    order_free_text(f[3])
                ],
            )
            .unwrap();
        }
        for text in [
            "Session::restore()",
            "Session::restore() father",
            "session",
            "session-session",
            "missing-restoration",
        ] {
            let q = LexicalQuery::new([text]);
            let memory = bm25(docs.iter().enumerate(), &q);
            let sqlite = conn
                .prepare(
                    "SELECT rowid, -bm25(docs,10,3,4,1) FROM docs
                WHERE docs MATCH ?1 ORDER BY bm25(docs,10,3,4,1)",
                )
                .unwrap()
                .query_map([q.fts5().unwrap()], |r| {
                    Ok((r.get::<_, i64>(0)? as usize, r.get::<_, f64>(1)?))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            assert_eq!(memory.len(), sqlite.len(), "{text}");
            for ((mi, ms), (si, ss)) in memory.iter().zip(&sqlite) {
                assert_eq!(mi, si, "{text}");
                assert!((ms - ss).abs() < 1e-12, "{text}: {ms} != {ss}");
            }
        }
    }

    #[test]
    fn contentless_reduced_detail_loses_bm25_and_phrases() {
        for detail in ["none", "column", "full"] {
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE docs USING fts5(name,body,
                content='',contentless_delete=1,detail={detail});
                INSERT INTO docs VALUES('alpha beta','');
                INSERT INTO docs VALUES('','alpha beta');
                INSERT INTO docs VALUES('','unrelated');
                INSERT INTO docs VALUES('','unrelated');
                INSERT INTO docs VALUES('','unrelated');"
            ))
            .unwrap();
            let scores = conn
                .prepare("SELECT -bm25(docs,10,1) FROM docs WHERE docs MATCH 'alpha'")
                .unwrap()
                .query_map([], |r| r.get::<_, f64>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            let phrase = conn.query_row(
                r#"SELECT count(*) FROM docs WHERE docs MATCH '"alpha beta"'"#,
                [],
                |r| r.get::<_, i64>(0),
            );
            let column = conn.query_row(
                "SELECT count(*) FROM docs WHERE docs MATCH 'name:alpha'",
                [],
                |r| r.get::<_, i64>(0),
            );
            if detail == "full" {
                assert!(scores[0] > scores[1] && scores[1] > 0.0);
                assert_eq!(phrase.unwrap(), 2);
            } else {
                assert_eq!(scores, [0.0, 0.0]);
                assert!(
                    phrase
                        .unwrap_err()
                        .to_string()
                        .contains("phrase queries are not supported")
                );
            }
            assert_eq!(column.is_err(), detail == "none");
        }
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
