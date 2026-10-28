use std::ops::Range;
use proc_macro2::Span;

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

}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::spanned::Spanned;

    #[test]
    fn parser_spans_preserve_original_bytes() {
        for prefix in ["", "\u{feff}", "#!/usr/bin/env rustx\n", "\u{feff}#!/é/rustx\r\n"] {
            for newline in ["\n", "\r\n"] {
                let item = format!("pub fn café() {{{newline}    let s = \"été\";{newline}}}");
                let source = format!("{prefix}// café{newline}{item}{newline}");
                let parsed = Parsed::new(&source).unwrap();
                assert_eq!(parsed.file.items.len(), 1);
                let range = parsed.range(parsed.file.items[0].span());
                assert_eq!(&source[range.clone()], item, "{source:?}: {range:?}");
                let syn::Item::Fn(function) = &parsed.file.items[0] else { panic!() };
                assert_eq!(&source[parsed.range(function.sig.ident.span())], "café");
                assert_eq!(&source[parsed.range(function.block.brace_token.span.open())], "{");
                assert_eq!(&source[parsed.range(function.block.brace_token.span.close())], "}");
            }
        }
    }
}
