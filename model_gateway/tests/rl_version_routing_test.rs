//! Gateway-level tests for RL M2: state-aware and version-aware routing.

mod common;

use std::collections::BTreeSet;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestRouterConfig,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn mock(port: u16) -> MockWorkerConfig {
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn ctx(policy: &str, workers: Vec<MockWorkerConfig>) -> AppTestContext {
    let mut config = TestRouterConfig::round_robin(0);
    config.rl.enabled = true;
    config.rl.control_timeout_secs = 5;
    config.rl.version_policy = policy.parse().expect("policy");
    AppTestContext::new_with_config(config, workers).await
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn json_of(resp: axum::response::Response) -> Value {
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes())
        .unwrap_or(Value::Null)
}

/// `(id, url)` of every registered worker, in `/v1/rl/workers` order.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn workers(app: &axum::Router) -> Vec<(String, String)> {
    let resp = app
        .clone()
        .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    json_of(resp).await["workers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| {
            (
                w["id"].as_str().unwrap().to_string(),
                w["url"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn post_json(app: &axum::Router, uri: &str, body: Value) -> axum::response::Response {
    app.clone()
        .oneshot(
            Request::post(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

/// Routed-worker URLs seen over `n` buffered `/generate` requests.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn routed_over(
    app: &axum::Router,
    n: usize,
    extra_header: Option<(&str, &str)>,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    for _ in 0..n {
        let mut req = Request::post("/generate").header("content-type", "application/json");
        if let Some((k, v)) = extra_header {
            req = req.header(k, v);
        }
        let body = json!({"text": "hi", "stream": false}).to_string();
        let resp = app
            .clone()
            .oneshot(req.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        seen.insert(
            resp.headers()["x-smg-routed-worker-id"]
                .to_str()
                .unwrap()
                .to_string(),
        );
    }
    seen
}

#[tokio::test]
async fn asleep_via_api_and_via_intercepted_release_get_no_dispatches() {
    let ctx = ctx("any", vec![mock(18921), mock(18922), mock(18923)]).await;
    let app = ctx.create_app();
    let ws = workers(&app).await;
    assert_eq!(ws.len(), 3);
    assert_eq!(
        routed_over(&app, 12, None).await.len(),
        3,
        "round robin over all three first"
    );

    // w0 asleep through the API, w1 through an intercepted release.
    let resp = post_json(
        &app,
        &format!("/v1/rl/workers/{}/state", ws[0].0),
        json!({"control": "asleep"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = post_json(
        &app,
        &format!(
            "/v1/rl/workers/{}/engine/release_memory_occupation",
            ws[1].0
        ),
        json!({"tags": ["weights"]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let routed = routed_over(&app, 12, None).await;
    assert_eq!(routed, BTreeSet::from([ws[2].1.clone()]));
    let listed = json_of(
        app.clone()
            .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    assert!(
        listed.to_string().matches("\"ready\"").count() >= 3,
        "still ready to the registry: {listed}"
    );

    // Wake them: API for w0, intercepted resume for w1.
    post_json(
        &app,
        &format!("/v1/rl/workers/{}/state", ws[0].0),
        json!({"control": "active"}),
    )
    .await;
    post_json(
        &app,
        &format!("/v1/rl/workers/{}/engine/resume_memory_occupation", ws[1].0),
        json!({"tags": ["weights"]}),
    )
    .await;
    assert_eq!(routed_over(&app, 12, None).await.len(), 3);
    ctx.shutdown().await;
}

#[tokio::test]
async fn rolling_update_under_latest_only_never_dispatches_to_a_stale_worker() {
    let ctx = ctx(
        "latest-only",
        vec![mock(18924), mock(18925), mock(18926), mock(18927)],
    )
    .await;
    let app = ctx.create_app();
    let ws = workers(&app).await;
    assert_eq!(
        routed_over(&app, 8, None).await.len(),
        4,
        "nobody versioned: nobody stale"
    );

    // Flip one at a time: API, then a proxied update_weight_version, then API.
    let resp = post_json(
        &app,
        &format!("/v1/rl/workers/{}/version", ws[0].0),
        json!({"weight_version": "2"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        routed_over(&app, 8, None).await,
        BTreeSet::from([ws[0].1.clone()])
    );

    let resp = post_json(
        &app,
        &format!("/v1/rl/workers/{}/engine/update_weight_version", ws[1].0),
        json!({"new_version": "2"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let routed = routed_over(&app, 8, None).await;
    assert!(
        routed.is_subset(&BTreeSet::from([ws[0].1.clone(), ws[1].1.clone()])),
        "{routed:?}"
    );

    let resp = post_json(
        &app,
        "/v1/rl/version?selector=engine%3Dsglang",
        json!({"weight_version": "2"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(json_of(resp).await["total"], 4);
    assert_eq!(routed_over(&app, 8, None).await.len(), 4);

    // A newer version on one worker makes the other three stale again.
    post_json(
        &app,
        &format!("/v1/rl/workers/{}/version", ws[3].0),
        json!({"weight_version": "3"}),
    )
    .await;
    assert_eq!(
        routed_over(&app, 8, None).await,
        BTreeSet::from([ws[3].1.clone()])
    );
    let listed = json_of(
        app.clone()
            .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    let sources: BTreeSet<&str> = listed["workers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["version_source"].as_str().unwrap())
        .collect();
    assert_eq!(sources, BTreeSet::from(["api"]));
    ctx.shutdown().await;
}

#[tokio::test]
async fn all_workers_ineligible_is_a_503() {
    let ctx = ctx("any", vec![mock(18930)]).await;
    let app = ctx.create_app();
    let ws = workers(&app).await;
    post_json(
        &app,
        &format!("/v1/rl/workers/{}/state", ws[0].0),
        json!({"control": "paused"}),
    )
    .await;
    let resp = app
        .clone()
        .oneshot(
            Request::post("/generate")
                .header("content-type", "application/json")
                .body(Body::from(json!({"text": "hi"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    ctx.shutdown().await;
}

#[tokio::test]
async fn flag_off_ignores_the_policy_header_and_stamps_nothing() {
    let mut config = TestRouterConfig::round_robin(0);
    config.rl.enabled = false;
    let ctx = AppTestContext::new_with_config(config, vec![mock(18931)]).await;
    let app = ctx.create_app();
    let resp = app
        .clone()
        .oneshot(
            Request::post("/generate")
                .header("content-type", "application/json")
                .header("x-smg-version-policy", "garbage")
                .body(Body::from(json!({"text": "hi"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get("x-smg-weight-version").is_none());
    assert!(resp.headers().get("x-smg-mixed-version").is_none());
    assert!(resp.headers().get("x-smg-routed-worker-id").is_some());
    assert!(!ctx.app_context.policy_registry.has_candidate_filter());
    ctx.shutdown().await;
}
