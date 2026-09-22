use openai_protocol::common::Tool;
use serde_json::json;
use tool_parser::{parsers::HyV4Parser, traits::ToolParser};

#[expect(clippy::unwrap_used, reason = "literal test fixture must deserialize")]
fn tools() -> Vec<Tool> {
    vec![serde_json::from_value(json!({"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"},"mixed":{"anyOf":[{"type":"string"},{"type":"boolean"}]}}}}})).unwrap()]
}

#[tokio::test]
async fn hy4_all_split_points_and_typed_arguments() {
    for suffix in ["", ":6124c78e", ":another-checkpoint"] {
        let text=format!("hello<tool_calls{suffix}><tool_call{suffix}>run<arg_key{suffix}>text</arg_key{suffix}><arg_value{suffix}>  中文🙂\"true\" &amp; </tool_call{suffix}>  </arg_value{suffix}><arg_key{suffix}>count</arg_key{suffix}><arg_value{suffix}>42</arg_value{suffix}><arg_key{suffix}>mixed</arg_key{suffix}><arg_value{suffix}>TRUE</arg_value{suffix}></tool_call{suffix}><tool_call{suffix}>run</tool_call{suffix}></tool_calls{suffix}>tail");
        let mut expected = None;
        for boundary in (0..=text.len()).filter(|i| text.is_char_boundary(*i)) {
            let mut p = HyV4Parser::new();
            let mut content = String::new();
            let mut calls = vec![];
            for chunk in [&text[..boundary], &text[boundary..]] {
                let r = p.parse_incremental(chunk, &tools()).await.unwrap();
                content.push_str(&r.normal_text);
                calls.extend(r.calls);
            }
            content.push_str(&p.take_unstreamed_normal_text());
            assert_eq!(content, "hellotail", "split {boundary}");
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].tool_index, 0);
            assert_eq!(calls[1].tool_index, 1);
            let args: serde_json::Value = serde_json::from_str(&calls[0].parameters).unwrap();
            assert_eq!(args["count"], 42);
            assert_eq!(args["mixed"], true);
            assert_eq!(
                args["text"],
                format!("  中文🙂\"true\" &amp; </tool_call{suffix}>  ")
            );
            expected = Some(args);
        }
        let (content, calls) = HyV4Parser::new()
            .parse_complete_with_tools(&text, &tools())
            .await
            .unwrap();
        assert_eq!(content, "hellotail");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            expected.unwrap()
        );
    }
}
#[tokio::test]
async fn hy4_text_truncation_and_reset() {
    let mut p = HyV4Parser::new();
    let r = p.parse_incremental("text <tool_ca", &[]).await.unwrap();
    assert_eq!(r.normal_text, "text ");
    assert_eq!(p.take_unstreamed_normal_text(), "<tool_ca");
    p.reset();
    let (_, calls) = p
        .parse_complete("<tool_calls><tool_call>run</tool_call></tool_calls>")
        .await
        .unwrap();
    assert_eq!(calls[0].function.arguments, "{}");
}

#[tokio::test]
async fn hy4_character_chunks_eof_and_factory() {
    let factory = tool_parser::factory::ParserFactory::new();
    let p = factory.get_parser("tencent/Hy4-preview-FP8").unwrap();
    assert!(p.has_tool_markers("<tool_calls:6124c78e>"));
    let text =
        "<tool_calls:6124c78e><tool_call:6124c78e>run</tool_call:6124c78e></tool_calls:6124c78e>";
    let mut parser = HyV4Parser::new();
    let mut calls = vec![];
    for ch in text.chars() {
        calls.extend(
            parser
                .parse_incremental(&ch.to_string(), &[])
                .await
                .unwrap()
                .calls,
        );
    }
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].parameters, "{}");
    parser.reset();
    let incomplete = "<tool_calls:6124c78e><tool_call:6124c78e>unfinished";
    assert!(parser
        .parse_incremental(incomplete, &[])
        .await
        .unwrap()
        .calls
        .is_empty());
    assert_eq!(parser.take_unstreamed_normal_text(), incomplete);
    assert!(parser.take_unstreamed_normal_text().is_empty());
}

#[tokio::test]
async fn hy4_malformed_is_not_a_partial_call_and_buffer_is_bounded() {
    let malformed="before<tool_calls><tool_call>run<arg_key>x</arg_key>missing_value</tool_call></tool_calls>after";
    let (content, calls) = HyV4Parser::new().parse_complete(malformed).await.unwrap();
    assert!(calls.is_empty());
    assert_eq!(content, malformed);
    let mut p = HyV4Parser::new();
    assert!(p
        .parse_incremental(&"x".repeat(4 * 1024 * 1024 + 1), &[])
        .await
        .is_err());
}
