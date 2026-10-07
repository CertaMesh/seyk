//! Splitting resources into independently searchable segments.
//!
//! Text is split along its own structure (Markdown headings, blank-line
//! paragraphs) into chunks of a few hundred tokens. Every segment keeps the
//! exact byte range it came from, so a hit always maps back to the source.

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Bump when segmentation output changes for the same input and config.
const TEXT_ALGORITHM_VERSION: u32 = 1;

/// How a text resource should be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextFormat {
    Markdown,
    Plain,
    Code,
}

/// Size limits for text segments, in characters.
///
/// Defaults are chosen for retrieval quality, not to fill the 8K context.
/// Gemma's tokenizer averages roughly four characters per token on English
/// prose, so the target of ~1,000 characters is about 250 tokens and the hard
/// maximum of 1,800 is about 450. A passage of that size usually covers one
/// idea, which keeps its vector specific. Long chunks average several ideas
/// together and a query matching one of them scores lower.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSegmenterConfig {
    /// Stop adding paragraphs once a chunk reaches this size.
    pub target_chars: usize,
    /// No chunk is longer than this. Longer paragraphs are split at sentences.
    pub max_chars: usize,
    /// When a section is split, a trailing paragraph up to this size is
    /// repeated at the start of the next chunk to keep local context.
    pub overlap_chars: usize,
}

impl Default for TextSegmenterConfig {
    fn default() -> Self {
        Self {
            target_chars: 1000,
            max_chars: 1800,
            overlap_chars: 400,
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

impl TextSegmenter {
    pub fn new(config: TextSegmenterConfig) -> Self {
        Self { config }
    }

    /// Identifies the algorithm and its parameters. Stored with cached vectors.
    pub fn version(&self) -> String {
        let c = &self.config;
        format!(
            "text-v{TEXT_ALGORITHM_VERSION}:t{}:m{}:o{}",
            c.target_chars, c.max_chars, c.overlap_chars
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
        let mut current: Vec<Block> = Vec::new();
        let mut current_len = 0;

        for block in blocks {
            let len = block.range.len();
            let only_headings = current.iter().all(|b| b.is_heading);

            if block.is_heading && !only_headings {
                // A heading always starts a new chunk.
                self.flush(source, &lines, &mut current, &mut out);
                current_len = 0;
            } else if !current.is_empty()
                && !only_headings
                && current_len + len > self.config.target_chars
            {
                let carry = current
                    .last()
                    .filter(|b| {
                        current.len() > 1
                            && !b.is_heading
                            && b.range.len() <= self.config.overlap_chars
                    })
                    .cloned()
                    .filter(|b| b.range.len() + len <= self.config.max_chars);
                self.flush(source, &lines, &mut current, &mut out);
                current_len = 0;
                if let Some(c) = carry {
                    current_len += c.range.len();
                    current.push(c);
                }
            }
            current_len += len;
            current.push(block);
        }
        self.flush(source, &lines, &mut current, &mut out);
        out
    }

    fn flush(
        &self,
        source: &str,
        lines: &LineIndex,
        current: &mut Vec<Block>,
        out: &mut Vec<TextSegment>,
    ) {
        if current.is_empty() {
            return;
        }
        let start = current.first().unwrap().range.start;
        let end = current.last().unwrap().range.end;
        let heading = current.last().unwrap().heading_path.clone();
        current.clear();
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
                let window_end = floor_char_boundary(source, start + max);
                let window = &source[start..window_end];
                // Prefer a sentence end, then any whitespace, past the halfway point.
                let min = window.len() / 2;
                let cut = rfind_sentence_end(window, min)
                    .or_else(|| {
                        window[min..]
                            .rfind(char::is_whitespace)
                            .map(|i| min + i + 1)
                    })
                    .unwrap_or(window.len());
                let piece_end = start + cut;
                out.push(trimmed_block(source, start..piece_end, &block));
                start = skip_whitespace(source, piece_end);
            }
            if start < block.range.end {
                out.push(trimmed_block(source, start..block.range.end, &block));
            }
        }
        out.retain(|b| !b.range.is_empty());
        out
    }
}

fn trimmed_block(source: &str, range: Range<usize>, like: &Block) -> Block {
    Block {
        range: trim_range(source, range),
        is_heading: false,
        heading_path: like.heading_path.clone(),
    }
}

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
        let src = "# Book\n\n## Father\n\nHe repeats his father's gestures.\n\n## Soup\n\nTomato soup needs basil.\n";
        let s = seg(src, TextFormat::Markdown);
        assert_eq!(s.len(), 2);
        assert!(
            s[0].text.starts_with("# Book"),
            "heading-only section merges forward"
        );
        assert!(s[0].text.contains("gestures"));
        assert_eq!(s[0].heading.as_deref(), Some("Book > Father"));
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

    #[test]
    fn crlf_line_numbers() {
        let src = "\r\na\r\n\r\nb\r\n";
        let s = seg(src, TextFormat::Plain);
        assert_eq!(s.len(), 1, "small paragraphs merge");
        assert_eq!((s[0].line_start, s[0].line_end), (2, 4));
        assert_eq!(s[0].text, "a\r\n\r\nb");
    }
}
