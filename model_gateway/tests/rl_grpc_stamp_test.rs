//! RL M2: the gRPC pipeline names the worker that served each response, and
//! reports the version the RL table holds for that engine.
//!
//! The routers are driven directly (no axum app), so the RL middleware never
//! runs here and `x-smg-weight-version` is not expected: what these tests pin
//! is the pipeline's own half of the contract -- the routed-worker header and
//! extension that the middleware keys on, and the live table version reaching
//! the body as `system_fingerprint`.

#[path = "common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::ChatCompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    middleware::TenantRequestMeta,
    routers::{common::header_utils::RoutedWorker, RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "rl-grpc-stamp-test-model";
/// Tokens the canned mock emits per request.
const OUTPUT_TOKENS: u32 = 3;
/// The version the RL table is told to hold for the registered engine. The
/// worker registers with no `weight_version` label, so the pipeline can only
/// report this by reading the table.
const LIVE_VERSION: &str = "11";

/// Spawn a canned mock gRPC worker in-process.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_worker() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: OUTPUT_TOKENS,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A regular-mode gRPC router over one mock worker, with the RL control plane
/// enabled so the app context carries an `RlState`.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_regular_router_with_rl(port: u16) -> (Box<dyn RouterTrait>, Arc<AppContext>) {
    let mut config = RouterConfig::builder()
        .grpc_connection()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.rl.enabled = true;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();

    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    let worker = BasicWorkerBuilder::new(worker_url(port))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    app_context
        .worker_registry
        .register(Arc::new(worker))
        .unwrap();

    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("regular gRPC router should build");
    (router, app_context)
}

fn worker_url(port: u16) -> String {
    format!("grpc://127.0.0.1:{port}")
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn chat_request(stream: bool) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("rl-grpc-stamp-test-tenant"))
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn read_body(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

/// A buffered gRPC chat response carries the routed-worker header and
/// extension, and reports the table's version rather than the registration
/// label (there is none) or the `"default"` fallback.
#[tokio::test]
async fn a_buffered_grpc_response_names_its_worker_and_reports_the_live_version() {
    let port = start_mock_grpc_worker().await;
    let (router, ctx) = build_regular_router_with_rl(port).await;
    let url = worker_url(port);
    let rl = ctx.rl.as_ref().expect("rl enabled");
    rl.table().set_version(
        &url,
        MODEL,
        smg_rl::Version::parse(LIVE_VERSION),
        smg_rl::VersionSource::Api,
    );

    let response = router
        .route_chat(None, &tenant(), chat_request(false), MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response.headers()["x-smg-routed-worker-id"],
        url.as_str(),
        "the buffered response must name the worker that served it"
    );
    assert!(
        response.extensions().get::<RoutedWorker>().is_some(),
        "the RL middleware keys on the RoutedWorker extension"
    );

    let body = read_body(response).await;
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["system_fingerprint"], LIVE_VERSION, "{body}");
}

/// The streamed variant of the same request: the SSE response is stamped too,
/// and its chunks carry the live version.
#[tokio::test]
async fn a_streamed_grpc_response_names_its_worker_and_reports_the_live_version() {
    let port = start_mock_grpc_worker().await;
    let (router, ctx) = build_regular_router_with_rl(port).await;
    let url = worker_url(port);
    let rl = ctx.rl.as_ref().expect("rl enabled");
    rl.table().set_version(
        &url,
        MODEL,
        smg_rl::Version::parse(LIVE_VERSION),
        smg_rl::VersionSource::Api,
    );

    let response = router
        .route_chat(None, &tenant(), chat_request(true), MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response.headers()["x-smg-routed-worker-id"],
        url.as_str(),
        "the SSE response must name the worker that served it"
    );
    assert!(
        response.extensions().get::<RoutedWorker>().is_some(),
        "the RL middleware keys on the RoutedWorker extension"
    );

    let body = read_body(response).await;
    assert!(
        body.contains(&format!("\"system_fingerprint\":\"{LIVE_VERSION}\"")),
        "streamed chunks must report the live version: {body}"
    );
}

/// With no version in the table the pipeline falls back to what the worker
/// reported at registration -- here nothing, so the `"default"` sentinel --
/// and the stamp is still applied.
#[tokio::test]
async fn an_unversioned_engine_still_stamps_and_falls_back_to_default() {
    let port = start_mock_grpc_worker().await;
    let (router, _ctx) = build_regular_router_with_rl(port).await;
    let url = worker_url(port);

    let response = router
        .route_chat(None, &tenant(), chat_request(false), MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(response.headers()["x-smg-routed-worker-id"], url.as_str());

    let body = read_body(response).await;
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["system_fingerprint"], "default", "{body}");
}
