//! Splitting resources into independently searchable segments.
//!
//! Text is split along its own structure (Markdown headings, blank-line
//! paragraphs) into chunks of a few hundred tokens. Every segment keeps the
//! exact byte range it came from, so a hit always maps back to the source.

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Bump when segmentation output changes for the same input and config.
const TEXT_ALGORITHM_VERSION: u32 = 2;

/// How a text resource should be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextFormat {
    Markdown,
    Plain,
    Code,
}

/// Size limits for text segments, in UTF-8 bytes (characters, for ASCII text).
///
/// Defaults are chosen for retrieval quality, not to fill the 8K context.
/// A vector that averages several ideas scores lower for a query matching only
/// one of them, so segments follow the author's own units: every paragraph of
/// at least `min_chars` is its own segment. Shorter fragments (labels, one-line
/// notes) are merged with their neighbours up to `target_chars`.
///
/// Gemma's tokenizer averages roughly four characters per token on English
/// prose: 60 characters is a short sentence, the 1,000-character target is
/// about 250 tokens and the 1,800-character maximum about 450.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSegmenterConfig {
    /// Paragraphs at least this long are never merged with each other.
    pub min_chars: usize,
    /// Stop merging short fragments once a chunk reaches this size.
    pub target_chars: usize,
    /// Longer paragraphs are split at sentences. A merged chunk can exceed this
    /// slightly, since headings and the gaps between paragraphs are not counted.
    pub max_chars: usize,
    /// When a long paragraph is cut at a sentence, its last sentence (if no
    /// longer than this) is repeated at the start of the next piece. There is
    /// no overlap across paragraph breaks the author chose.
    pub overlap_chars: usize,
}

impl Default for TextSegmenterConfig {
    fn default() -> Self {
        Self {
            min_chars: 60,
            target_chars: 1000,
            max_chars: 1800,
            overlap_chars: 300,
        }
    }
}

/// A searchable slice of a text resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSegment {
    /// Byte range into the source. `&source[byte_range]` is exactly `text`.
    pub byte_range: Range<usize>,
    /// 1-based, inclusive line numbers.
    pub line_start: u64,
    pub line_end: u64,
    /// Markdown heading path, for example `Part One > The Kitchen`.
    pub heading: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextSegmenter {
    pub config: TextSegmenterConfig,
}

#[derive(Debug, Clone)]
struct Block {
    range: Range<usize>,
    is_heading: bool,
    heading_path: Option<String>,
}

/// The chunk being built: one contiguous source span, gaps included.
struct Pending {
    range: Range<usize>,
    /// Heading path of the last block added.
    heading: Option<String>,
    /// Bytes of non-heading text, so 0 means the chunk is only headings.
    body_len: usize,
}

impl TextSegmenter {
    pub fn new(config: TextSegmenterConfig) -> Self {
        Self { config }
    }

    /// Identifies the algorithm and its parameters. Stored with cached vectors.
    pub fn version(&self) -> String {
        let c = &self.config;
        format!(
            "text-v{TEXT_ALGORITHM_VERSION}:n{}:t{}:m{}:o{}",
            c.min_chars, c.target_chars, c.max_chars, c.overlap_chars
        )
    }

    pub fn segment(&self, source: &str, format: TextFormat) -> Vec<TextSegment> {
        let blocks = match format {
            TextFormat::Markdown => markdown_blocks(source),
            TextFormat::Plain | TextFormat::Code => paragraph_blocks(source),
        };
        let blocks = self.split_oversized(source, blocks);
        let lines = LineIndex::new(source);

        let mut out = Vec::new();
        let mut current: Option<Pending> = None;

        for block in blocks {
            let len = block.range.len();
            // Every non-heading block is nonempty, so no body means only headings.
            let body_len = current.as_ref().map_or(0, |c| c.body_len);
            let only_headings = body_len == 0;
            let min = self.config.min_chars;

            // A heading always starts a new chunk; a heading-only chunk merges forward.
            let new_section = block.is_heading && !only_headings;
            // Two substantial paragraphs are separate ideas.
            let both_substantial = !block.is_heading && body_len >= min && len >= min;
            let too_big = !only_headings && body_len + len > self.config.target_chars;
            if new_section || both_substantial || too_big {
                self.flush(source, &lines, current.take(), &mut out);
            }
            let added = if block.is_heading { 0 } else { len };
            match &mut current {
                Some(c) => {
                    c.range.end = block.range.end;
                    c.heading = block.heading_path;
                    c.body_len += added;
                }
                None => {
                    current = Some(Pending {
                        range: block.range,
                        heading: block.heading_path,
                        body_len: added,
                    })
                }
            }
        }
        self.flush(source, &lines, current, &mut out);
        out
    }

    fn flush(
        &self,
        source: &str,
        lines: &LineIndex,
        pending: Option<Pending>,
        out: &mut Vec<TextSegment>,
    ) {
        let Some(Pending {
            range: Range { start, end },
            heading,
            ..
        }) = pending
        else {
            return;
        };
        let text = &source[start..end];
        if text.trim().is_empty() {
            return;
        }
        out.push(TextSegment {
            byte_range: start..end,
            line_start: lines.line_of(start),
            line_end: lines.line_of(end.saturating_sub(1).max(start)),
            heading,
            text: text.to_string(),
        });
    }

    /// Split any block longer than `max_chars` at sentence boundaries.
    fn split_oversized(&self, source: &str, blocks: Vec<Block>) -> Vec<Block> {
        let max = self.config.max_chars;
        let mut out = Vec::with_capacity(blocks.len());
        for block in blocks {
            if block.range.len() <= max {
                out.push(block);
                continue;
            }
            let mut start = block.range.start;
            while block.range.end - start > max {
                // Keep whitespace skipping inside this block, with absolute offsets.
                let (piece_end, next) = self.split_point(&source[..block.range.end], start);
                out.push(trimmed_block(source, start..piece_end, &block));
                start = next;
            }
            if start < block.range.end {
                out.push(trimmed_block(source, start..block.range.end, &block));
            }
        }
        out.retain(|b| !b.range.is_empty());
        out
    }
}

impl TextSegmenter {
    /// Where the piece starting at `start` ends, and where the next one begins.
    ///
    /// Budgets are bytes, so every offset is snapped to a char boundary, and the
    /// next start is always past `start`. A budget too small for even one char
    /// still takes that whole char, so the caller always makes progress.
    fn split_point(&self, source: &str, start: usize) -> (usize, usize) {
        let mut window_end = floor_char_boundary(source, start + self.config.max_chars);
        if window_end == start {
            window_end += source[start..].chars().next().map_or(1, char::len_utf8);
        }
        let window = &source[start..window_end];
        // Prefer a sentence end, then any whitespace, past the halfway point.
        let half = window.len() / 2;
        let cut = rfind_sentence_end(window, half)
            .or_else(|| {
                window
                    .char_indices()
                    .rev()
                    .take_while(|&(i, _)| i >= half)
                    .find(|&(_, c)| c.is_whitespace())
                    .map(|(i, c)| i + c.len_utf8())
            })
            .unwrap_or(window.len());
        let piece_end = start + cut;
        // Repeat the piece's last sentence so a thought cut in half is still
        // whole in one of the two pieces. The piece is trimmed, so its own final
        // punctuation is never followed by whitespace and cannot match.
        let piece = source[start..piece_end].trim_end();
        let next = rfind_sentence_end(piece, 0)
            .map(|i| skip_whitespace(source, start + i))
            .filter(|&s| s > start && piece_end - s <= self.config.overlap_chars)
            .unwrap_or_else(|| skip_whitespace(source, piece_end));
        (piece_end, next)
    }
}

fn trimmed_block(source: &str, range: Range<usize>, like: &Block) -> Block {
    Block {
        range: trim_range(source, range),
        is_heading: false,
        heading_path: like.heading_path.clone(),
    }
}

/// Offset just past the last ASCII sentence end at or after `min`. ASCII, so
/// always a char boundary.
fn rfind_sentence_end(window: &str, min: usize) -> Option<usize> {
    let bytes = window.as_bytes();
    (min..bytes.len().saturating_sub(1))
        .rev()
        .find(|&i| {
            matches!(bytes[i], b'.' | b'!' | b'?' | b'\n') && bytes[i + 1].is_ascii_whitespace()
        })
        .map(|i| i + 1)
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn skip_whitespace(s: &str, mut i: usize) -> usize {
    while i < s.len() {
        let c = s[i..].chars().next().unwrap();
        if !c.is_whitespace() {
            break;
        }
        i += c.len_utf8();
    }
    i
}

fn trim_range(s: &str, r: Range<usize>) -> Range<usize> {
    let slice = &s[r.clone()];
    let lead = slice.len() - slice.trim_start().len();
    let trail = slice.len() - slice.trim_end().len();
    if lead == slice.len() {
        return r.start..r.start;
    }
    (r.start + lead)..(r.end - trail)
}

/// Lines with their byte range, excluding the newline.
fn lines_with_ranges(source: &str) -> impl Iterator<Item = (Range<usize>, &str)> {
    let mut pos = 0;
    source.split_inclusive('\n').map(move |raw| {
        let start = pos;
        pos += raw.len();
        let line = raw.trim_end_matches(['\n', '\r']);
        (start..start + line.len(), line)
    })
}

/// Blank-line separated paragraphs.
fn paragraph_blocks(source: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut para: Option<Range<usize>> = None;
    for (range, line) in lines_with_ranges(source) {
        if line.trim().is_empty() {
            if let Some(p) = para.take() {
                blocks.push(Block {
                    range: p,
                    is_heading: false,
                    heading_path: None,
                });
            }
        } else {
            para = Some(match para {
                Some(p) => p.start..range.end,
                None => range,
            });
        }
    }
    if let Some(p) = para {
        blocks.push(Block {
            range: p,
            is_heading: false,
            heading_path: None,
        });
    }
    for b in &mut blocks {
        b.range = trim_range(source, b.range.clone());
    }
    blocks
}

/// Markdown paragraphs, ATX headings and fenced code blocks.
fn markdown_blocks(source: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut headings: Vec<(usize, String)> = Vec::new();
    let mut para: Option<Range<usize>> = None;
    let mut fence: Option<(String, Range<usize>)> = None;

    let path = |headings: &[(usize, String)]| {
        (!headings.is_empty()).then(|| {
            headings
                .iter()
                .map(|(_, t)| t.as_str())
                .collect::<Vec<_>>()
                .join(" > ")
        })
    };

    for (range, line) in lines_with_ranges(source) {
        let trimmed = line.trim_start();

        if let Some((marker, frange)) = &mut fence {
            frange.end = range.end;
            if trimmed.starts_with(marker.as_str()) {
                blocks.push(Block {
                    range: frange.clone(),
                    is_heading: false,
                    heading_path: path(&headings),
                });
                fence = None;
            }
            continue;
        }

        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if let Some(p) = para.take() {
                blocks.push(Block {
                    range: p,
                    is_heading: false,
                    heading_path: path(&headings),
                });
            }
            fence = Some((trimmed[..3].to_string(), range));
            continue;
        }

        if let Some((level, title)) = atx_heading(line) {
            if let Some(p) = para.take() {
                blocks.push(Block {
                    range: p,
                    is_heading: false,
                    heading_path: path(&headings),
                });
            }
            headings.retain(|(l, _)| *l < level);
            headings.push((level, title));
            blocks.push(Block {
                range,
                is_heading: true,
                heading_path: path(&headings),
            });
            continue;
        }

        if line.trim().is_empty() {
            if let Some(p) = para.take() {
                blocks.push(Block {
                    range: p,
                    is_heading: false,
                    heading_path: path(&headings),
                });
            }
        } else {
            para = Some(match para {
                Some(p) => p.start..range.end,
                None => range,
            });
        }
    }
    if let Some((_, frange)) = fence {
        blocks.push(Block {
            range: frange,
            is_heading: false,
            heading_path: path(&headings),
        });
    }
    if let Some(p) = para {
        blocks.push(Block {
            range: p,
            is_heading: false,
            heading_path: path(&headings),
        });
    }
    for b in &mut blocks {
        b.range = trim_range(source, b.range.clone());
    }
    blocks.retain(|b| !b.range.is_empty());
    blocks
}

fn atx_heading(line: &str) -> Option<(usize, String)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let level = rest.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let after = &rest[level..];
    if !after.is_empty() && !after.starts_with([' ', '\t']) {
        return None;
    }
    let title = after.trim().trim_end_matches('#').trim().to_string();
    Some((level, title))
}

/// Maps byte offsets to 1-based line numbers.
pub(crate) struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    pub(crate) fn new(source: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        Self { starts }
    }

    pub(crate) fn line_of(&self, byte: usize) -> u64 {
        (self.starts.partition_point(|&s| s <= byte)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(src: &str, fmt: TextFormat) -> Vec<TextSegment> {
        TextSegmenter::default().segment(src, fmt)
    }

    #[test]
    fn locators_map_back_to_source() {
        let src = "# Title\n\nFirst para line one.\nline two.\n\n## Sub\n\nSecond para.\n";
        for s in seg(src, TextFormat::Markdown) {
            assert_eq!(&src[s.byte_range.clone()], s.text);
            let lines: Vec<&str> = src.lines().collect();
            assert!(!lines[(s.line_start - 1) as usize].is_empty());
            assert!(s.text.ends_with(lines[(s.line_end - 1) as usize].trim()));
        }
    }

    #[test]
    fn headings_stay_with_their_content() {
        let src = "# Book\n\n## Garden\n\nShe waters the beds at dawn.\n\n## Soup\n\nTomato soup needs basil.\n";
        let s = seg(src, TextFormat::Markdown);
        assert_eq!(s.len(), 2);
        assert!(
            s[0].text.starts_with("# Book"),
            "heading-only section merges forward"
        );
        assert!(s[0].text.contains("dawn"));
        assert_eq!(s[0].heading.as_deref(), Some("Book > Garden"));
        assert!(s[1].text.starts_with("## Soup"));
        assert_eq!(s[1].heading.as_deref(), Some("Book > Soup"));
        assert_eq!(s[1].line_start, 7);
        assert_eq!(s[1].line_end, 9);
    }

    #[test]
    fn boundaries_are_stable() {
        let para = "This sentence is about memory and how it fades over years. ".repeat(8);
        let src = format!("# A\n\n{para}\n\n{para}\n\n{para}\n\n# B\n\n{para}\n");
        let a = seg(&src, TextFormat::Markdown);
        let b = seg(&src, TextFormat::Markdown);
        assert_eq!(a, b);
        assert!(a.len() >= 3);
        for s in &a {
            assert!(
                s.text.len() <= TextSegmenterConfig::default().max_chars,
                "{}",
                s.text.len()
            );
        }
        // Editing section B must not move section A's boundaries.
        let edited = src.replace("# B\n\n", "# B\n\nA new opening line.\n\n");
        let c = seg(&edited, TextFormat::Markdown);
        let a_only: Vec<_> = a
            .iter()
            .filter(|s| s.heading.as_deref() == Some("A"))
            .collect();
        let c_only: Vec<_> = c
            .iter()
            .filter(|s| s.heading.as_deref() == Some("A"))
            .collect();
        assert_eq!(a_only, c_only);
    }

    #[test]
    fn long_paragraph_is_split_at_sentences() {
        let src = "One sentence about grief as a weight in the chest. ".repeat(80);
        let s = seg(&src, TextFormat::Plain);
        assert!(s.len() > 1);
        for piece in &s {
            assert!(piece.text.len() <= 1800);
            assert!(
                piece.text.ends_with('.'),
                "split at sentence end: {:?}",
                &piece.text[piece.text.len() - 10..]
            );
            assert_eq!(&src[piece.byte_range.clone()], piece.text);
        }
    }

    #[test]
    fn substantial_paragraphs_stay_separate_and_fragments_merge() {
        let src = "Grief: after the funeral her chest felt packed with wet sand.\n\n\
                   Memory: she is certain the argument happened in winter, not summer.\n\n\
                   TODO\n\n\
                   Inheritance: a temper, a way of folding shirts, the silence at dinner.\n";
        let s = seg(src, TextFormat::Plain);
        let texts: Vec<&str> = s.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(texts.len(), 3, "{texts:#?}");
        assert!(texts[0].starts_with("Grief"));
        assert!(
            texts[1].starts_with("Memory") && texts[1].ends_with("TODO"),
            "fragment joins its neighbour"
        );
        assert!(texts[2].starts_with("Inheritance"));
        assert_eq!((s[1].line_start, s[1].line_end), (3, 5));
    }

    #[test]
    fn split_paragraphs_overlap_by_one_sentence() {
        let src = (0..60)
            .map(|i| format!("Sentence number {i} is about the long walk home."))
            .collect::<Vec<_>>()
            .join(" ");
        let s = seg(&src, TextFormat::Plain);
        assert!(s.len() > 1);
        for pair in s.windows(2) {
            let last_sentence = pair[0].text.rsplit(". ").next().unwrap();
            assert!(
                pair[1]
                    .text
                    .starts_with(last_sentence.trim_end_matches('.')),
                "{:?}",
                &pair[1].text[..40]
            );
            assert!(pair[1].byte_range.start < pair[0].byte_range.end);
        }
    }

    #[test]
    fn fenced_code_is_not_parsed_as_headings() {
        let src = "# Real\n\n```\n# not a heading\n```\n";
        let s = seg(src, TextFormat::Markdown);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].heading.as_deref(), Some("Real"));
    }

    #[test]
    fn multibyte_text_is_safe() {
        let src = "Ünïcödé — „Zitat“ ✓. ".repeat(200);
        for s in seg(&src, TextFormat::Plain) {
            assert_eq!(&src[s.byte_range.clone()], s.text);
        }
    }

    /// Every piece is nonempty, maps back to its source slice, and the pieces
    /// cover the text in order.
    fn assert_well_formed(src: &str, segs: &[TextSegment]) {
        assert!(!segs.is_empty() || src.trim().is_empty());
        let mut last_start = None;
        for s in segs {
            assert!(!s.text.is_empty());
            assert_eq!(&src[s.byte_range.clone()], s.text);
            assert!(last_start < Some(s.byte_range.start), "pieces must advance");
            last_start = Some(s.byte_range.start);
        }
    }

    #[test]
    fn long_runs_without_punctuation_split_on_char_boundaries() {
        // 2, 3 and 4 byte chars with no sentence end or ASCII space to cut at;
        // these used to panic by slicing inside a char.
        for unit in ["é", "日本語", "─", "🙂"] {
            let src = unit.repeat(3000 / unit.len() + 1);
            assert_well_formed(&src, &seg(&src, TextFormat::Plain));
        }
        // Multibyte whitespace (NBSP, em space) as the only cut points.
        for ws in ['\u{a0}', '\u{2003}'] {
            let src = format!("wörd{ws}").repeat(800);
            assert_well_formed(&src, &seg(&src, TextFormat::Plain));
        }
    }

    #[test]
    fn tiny_budgets_still_make_progress() {
        let src = "Ab é 日. Cd ─ 🙂! Ef gh.";
        for max_chars in 0..6 {
            let s = TextSegmenter::new(TextSegmenterConfig {
                min_chars: 0,
                target_chars: max_chars,
                max_chars,
                overlap_chars: 2,
            })
            .segment(src, TextFormat::Plain);
            assert_well_formed(src, &s);
        }
    }

    #[test]
    fn tiny_budget_does_not_skip_past_paragraph_end() {
        let segmenter = TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 1,
            max_chars: 1,
            overlap_chars: 0,
        });
        let segments = segmenter.segment("é\n\nx", TextFormat::Plain);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].byte_range, 0..2);
        assert_eq!(segments[0].text, "é");
        assert_eq!(segments[1].byte_range, 4..5);
        assert_eq!(segments[1].text, "x");
        assert_eq!((segments[0].line_start, segments[1].line_start), (1, 3));
    }

    fn blocks_for(source: &str, format: TextFormat) -> Vec<Block> {
        match format {
            TextFormat::Markdown => markdown_blocks(source),
            TextFormat::Plain | TextFormat::Code => paragraph_blocks(source),
        }
    }

    fn assert_ranges_cover(source: &str, ranges: &[Range<usize>]) {
        for range in ranges {
            assert!(range.start < range.end);
            assert!(range.end <= source.len());
            assert!(source.is_char_boundary(range.start));
            assert!(source.is_char_boundary(range.end));
        }
        for pair in ranges.windows(2) {
            assert!(pair[0].start < pair[1].start, "starts must advance");
            assert!(pair[0].end <= pair[1].end, "ends must not retreat");
        }
        // Whitespace may be trimmed or skipped; every other character must be
        // represented, including when adjacent ranges deliberately overlap.
        for (offset, ch) in source.char_indices() {
            assert!(ch.is_whitespace() || ranges.iter().any(|range| range.contains(&offset)));
        }
    }

    #[test]
    fn tiny_budgets_stay_in_blocks_and_cover_multibyte_text() {
        let sources = [
            "é\n\nx",
            "日\n\n🙂\n\né\t \n\n",
            " \té\u{2003}\n\n日\u{a0}\r\n\r\n🙂 \t",
            "Ab é 日. Cd ─ 🙂! Ef gh.\n\n é \n\n",
            "a. b. c. d. é. 日. 🙂.\n\nx\n",
            "# é\n\n日\n\n## 🙂\n\né\t\n",
            "# 日\n\n```\né\n🙂\n```\n\n日\t\n",
            "\n\t\u{2003}\n",
            "",
        ];
        for source in sources {
            for format in [TextFormat::Plain, TextFormat::Markdown] {
                for max_chars in 0..=5 {
                    for overlap_chars in [0, 2, 5] {
                        let segmenter = TextSegmenter::new(TextSegmenterConfig {
                            min_chars: 0,
                            target_chars: max_chars,
                            max_chars,
                            overlap_chars,
                        });
                        // Inspect each original block independently: public
                        // segments can intentionally merge headings forward.
                        for block in blocks_for(source, format) {
                            let pieces = segmenter.split_oversized(source, vec![block.clone()]);
                            for piece in &pieces {
                                assert!(piece.range.start >= block.range.start);
                                assert!(piece.range.end <= block.range.end);
                            }
                            let relative_ranges: Vec<_> = pieces
                                .iter()
                                .map(|piece| {
                                    piece.range.start - block.range.start
                                        ..piece.range.end - block.range.start
                                })
                                .collect();
                            assert_ranges_cover(&source[block.range], &relative_ranges);
                        }
                        let segments = segmenter.segment(source, format);
                        assert_well_formed(source, &segments);
                        let ranges: Vec<_> =
                            segments.iter().map(|s| s.byte_range.clone()).collect();
                        assert_ranges_cover(source, &ranges);
                    }
                }
            }
        }
    }

    // Frozen pre-fix split_oversized. The helpers and the public segment
    // assembly are unchanged; comparing every resulting Block field proves
    // identical inputs to that assembly, including heading and byte metadata.
    fn old_split_oversized(
        segmenter: &TextSegmenter,
        source: &str,
        blocks: Vec<Block>,
    ) -> Vec<Block> {
        let max = segmenter.config.max_chars;
        let mut out = Vec::with_capacity(blocks.len());
        for block in blocks {
            if block.range.len() <= max {
                out.push(block);
                continue;
            }
            let mut start = block.range.start;
            while block.range.end - start > max {
                let (piece_end, next) = segmenter.split_point(source, start);
                out.push(trimmed_block(source, start..piece_end, &block));
                start = next;
            }
            if start < block.range.end {
                out.push(trimmed_block(source, start..block.range.end, &block));
            }
        }
        out.retain(|b| !b.range.is_empty());
        out
    }

    #[test]
    fn successful_splits_match_the_previous_algorithm() {
        let paragraphs = [
            "é",
            "日",
            "🙂",
            "a. b. é! 日? 🙂.",
            "# Héading",
            "```\né\n```",
        ];
        let mut compared = 0;
        let mut formerly_panicked = 0;
        let signature = |blocks: Vec<Block>| {
            blocks
                .into_iter()
                .map(|block| (block.range, block.is_heading, block.heading_path))
                .collect::<Vec<_>>()
        };
        for first in paragraphs {
            for second in paragraphs {
                for separator in ["\n\n", "\r\n\r\n", "\n \t\n", "\n\u{2003}\n"] {
                    let source = format!("{first}{separator}{second}\t \n\n");
                    for format in [TextFormat::Plain, TextFormat::Markdown, TextFormat::Code] {
                        for max_chars in [0, 1, 2, 3, 4, 5, 6, 12, 64, 1800] {
                            for overlap_chars in [0, 2, 300] {
                                let segmenter = TextSegmenter::new(TextSegmenterConfig {
                                    max_chars,
                                    overlap_chars,
                                    ..TextSegmenterConfig::default()
                                });
                                let blocks = blocks_for(&source, format);
                                let new =
                                    signature(segmenter.split_oversized(&source, blocks.clone()));
                                match std::panic::catch_unwind(|| {
                                    old_split_oversized(&segmenter, &source, blocks)
                                }) {
                                    Ok(old) => {
                                        assert_eq!(
                                            new,
                                            signature(old),
                                            "source={source:?}, format={format:?}, config={:?}",
                                            segmenter.config
                                        );
                                        compared += 1;
                                    }
                                    Err(_) => formerly_panicked += 1,
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(compared > 0);
        assert!(formerly_panicked > 0, "the table must exercise the old bug");
        assert_eq!(compared + formerly_panicked, 12_960);
        println!("{compared} identical successful splits; {formerly_panicked} former panics fixed");
    }

    #[test]
    fn crlf_line_numbers() {
        let src = "\r\na\r\n\r\nb\r\n";
        let s = seg(src, TextFormat::Plain);
        assert_eq!(s.len(), 1, "small paragraphs merge");
        assert_eq!((s[0].line_start, s[0].line_end), (2, 4));
        assert_eq!(s[0].text, "a\r\n\r\nb");
    }
}
