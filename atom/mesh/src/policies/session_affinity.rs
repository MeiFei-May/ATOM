//! Session table that can wrap any load-balancing policy.
//!
//! The first request for a session is placed by the inner policy. Later
//! requests stay on that worker until the idle TTL expires or the worker is
//! unusable. `dp_sticky` uses [`MinSessionPolicy`] for that first placement
//! (fewest live sessions, then load). Every other policy is used unchanged.
//!
//! `hard` always returns the bound worker. `soft` does too, unless the inner
//! policy opts in via [`LoadBalancingPolicy::overrides_session_affinity`].

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;

use super::{
    get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo,
    NO_HEALTHY_WORKERS,
};
use crate::{
    config::{SessionAffinityConfig, SessionAffinityMode, SessionParentMode},
    core::Worker,
    observability::metrics::{
        metrics_labels::{
            AFFINITY_EXPIRED, AFFINITY_HIT, AFFINITY_NEW, AFFINITY_REBIND,
            WORKER_CHANGE_RANK_MAPPING,
        },
        MeshMetrics,
    },
    routers::comm::header_utils::{
        extract_first_header, extract_header_value, PARENT_SESSION_HEADER,
    },
};

const NO_SESSION: &str = "no_session";
const SWEEP_INTERVAL: Duration = Duration::from_secs(30);

/// Which session table produced a decision. Prefill and decode are separate
/// counters so a healthy decode table cannot hide prefill misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionAffinityScope {
    Regular,
    Prefill,
    Decode,
}

impl SessionAffinityScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct WorkerIdentity {
    url: String,
    dp_rank: Option<usize>,
}

impl WorkerIdentity {
    fn from_worker(worker: &Arc<dyn Worker>) -> Self {
        Self {
            url: worker.url().to_string(),
            dp_rank: worker.dp_rank(),
        }
    }
}

#[derive(Debug)]
struct StickyAssignment {
    worker: WorkerIdentity,
    last_access: Instant,
    /// Where the session's last request was sent, after DP rank mapping.
    dispatched: Option<WorkerIdentity>,
    /// Why the session was last reassigned, until a request is sent.
    reassigned: Option<&'static str>,
}

#[derive(Debug, Default)]
pub(crate) struct SessionTable {
    assignments: DashMap<String, StickyAssignment>,
}

impl SessionTable {
    /// Bind `key` to `worker`. A rebound session keeps where its last request
    /// was sent, so the next dispatch can tell whether the worker changed.
    fn assign(
        &self,
        key: String,
        worker: WorkerIdentity,
        now: Instant,
        reassigned: Option<&'static str>,
    ) {
        let mut entry = self
            .assignments
            .entry(key)
            .or_insert_with(|| StickyAssignment {
                worker: worker.clone(),
                last_access: now,
                dispatched: None,
                reassigned: None,
            });
        entry.worker = worker;
        entry.last_access = now;
        entry.reassigned = reassigned;
    }

    fn touch(&self, key: &str, now: Instant) {
        if let Some(mut entry) = self.assignments.get_mut(key) {
            entry.last_access = now;
        }
    }

    /// Record where a request of `key` was sent. Returns why the session moved
    /// when this worker differs from the one its previous request was sent to.
    fn record_dispatch(&self, key: &str, worker: WorkerIdentity) -> Option<&'static str> {
        let mut entry = self.assignments.get_mut(key)?;
        // Without a reassignment, only rank mapping can move a pinned session.
        let reason = entry
            .reassigned
            .take()
            .unwrap_or(WORKER_CHANGE_RANK_MAPPING);
        let moved = entry
            .dispatched
            .as_ref()
            .is_some_and(|previous| *previous != worker);
        entry.dispatched = Some(worker);
        moved.then_some(reason)
    }

    fn live_counts(&self, now: Instant, ttl: Duration) -> HashMap<WorkerIdentity, usize> {
        let mut live = HashMap::new();
        for entry in self.assignments.iter() {
            if now.saturating_duration_since(entry.last_access) <= ttl {
                *live.entry(entry.worker.clone()).or_insert(0) += 1;
            }
        }
        live
    }

    /// Drop idle bindings. `keep` is left in place so the request being routed
    /// can still tell a new session from an expired one.
    fn retain_live(&self, now: Instant, ttl: Duration, keep: Option<&str>) {
        self.assignments.retain(|key, assignment| {
            keep == Some(key.as_str())
                || now.saturating_duration_since(assignment.last_access) <= ttl
        });
    }
}

/// First placement for `dp_sticky`: fewest non-expired sessions, then load.
/// Requests without a session key use [`LoadBalancingPolicy::select_with_decision`],
/// which is minimum load.
#[derive(Debug)]
pub(crate) struct MinSessionPolicy {
    table: Arc<SessionTable>,
    ttl: Duration,
}

impl MinSessionPolicy {
    pub(crate) fn new(table: Arc<SessionTable>, ttl: Duration) -> Self {
        Self { table, ttl }
    }

    pub(crate) fn select_fewest(&self, workers: &[Arc<dyn Worker>]) -> Option<usize> {
        let now = Instant::now();
        let live = self.table.live_counts(now, self.ttl);
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| {
                let identity = WorkerIdentity::from_worker(&workers[index]);
                (
                    live.get(&identity).copied().unwrap_or(0),
                    workers[index].load(),
                )
            })
    }

    fn select_low_load(workers: &[Arc<dyn Worker>]) -> Option<usize> {
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| workers[index].load())
    }
}

#[async_trait]
impl LoadBalancingPolicy for MinSessionPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let selected = Self::select_low_load(workers);
        let branch = if selected.is_some() {
            NO_SESSION
        } else {
            NO_HEALTHY_WORKERS
        };
        (selected, RouteDecision::branch(branch))
    }

    fn name(&self) -> &'static str {
        "dp_sticky"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct SessionAffinityPolicy {
    config: SessionAffinityConfig,
    ttl: Duration,
    table: Arc<SessionTable>,
    inner: Arc<dyn LoadBalancingPolicy>,
    scope: SessionAffinityScope,
    assign_lock: tokio::sync::Mutex<()>,
    last_sweep: std::sync::Mutex<Instant>,
    results: [AtomicU64; 4],
}

impl std::fmt::Debug for SessionAffinityPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionAffinityPolicy")
            .field("config", &self.config)
            .field("inner", &self.inner.name())
            .field("scope", &self.scope)
            .finish()
    }
}

impl SessionAffinityPolicy {
    #[cfg(test)]
    pub(crate) fn around(
        config: SessionAffinityConfig,
        inner: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        Self::around_scoped(config, inner, SessionAffinityScope::Regular)
    }

    pub(crate) fn around_scoped(
        config: SessionAffinityConfig,
        inner: Arc<dyn LoadBalancingPolicy>,
        scope: SessionAffinityScope,
    ) -> Self {
        let ttl = Duration::from_secs(config.ttl_secs);
        Self::new(config, ttl, Arc::new(SessionTable::default()), inner, scope)
    }

    /// `dp_sticky`: the session table and the fewest-session placement share one table.
    #[cfg(test)]
    pub(crate) fn dp_sticky(config: SessionAffinityConfig) -> Self {
        Self::dp_sticky_scoped(config, SessionAffinityScope::Regular)
    }

    pub(crate) fn dp_sticky_scoped(
        config: SessionAffinityConfig,
        scope: SessionAffinityScope,
    ) -> Self {
        let ttl = Duration::from_secs(config.ttl_secs);
        let table = Arc::new(SessionTable::default());
        let inner = Arc::new(MinSessionPolicy::new(Arc::clone(&table), ttl));
        Self::new(config, ttl, table, inner, scope)
    }

    fn new(
        config: SessionAffinityConfig,
        ttl: Duration,
        table: Arc<SessionTable>,
        inner: Arc<dyn LoadBalancingPolicy>,
        scope: SessionAffinityScope,
    ) -> Self {
        Self {
            config,
            ttl,
            table,
            inner,
            scope,
            assign_lock: tokio::sync::Mutex::new(()),
            last_sweep: std::sync::Mutex::new(
                Instant::now()
                    .checked_sub(SWEEP_INTERVAL)
                    .unwrap_or_else(Instant::now),
            ),
            results: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    pub(crate) fn inner(&self) -> &dyn LoadBalancingPolicy {
        self.inner.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn result_count(&self, result: &str) -> u64 {
        self.results[result_index(result)].load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn scope_label(&self) -> &'static str {
        self.scope.as_str()
    }

    #[cfg(test)]
    pub(crate) fn table_len(&self) -> usize {
        self.table.assignments.len()
    }

    #[cfg(test)]
    fn has_session(&self, key: &str) -> bool {
        self.table.assignments.contains_key(key)
    }

    #[cfg(test)]
    fn expire_sweep_clock(&self) {
        let mut last = self
            .last_sweep
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *last = Instant::now() - SWEEP_INTERVAL - Duration::from_secs(1);
    }

    #[cfg(test)]
    pub(crate) fn set_last_access(&self, key: &str, when: Instant) {
        if let Some(mut entry) = self.table.assignments.get_mut(key) {
            entry.last_access = when;
        }
    }

    fn binding_key(&self, headers: Option<&http::HeaderMap>) -> Option<String> {
        let names: Vec<&str> = self
            .config
            .session_id_headers
            .iter()
            .map(String::as_str)
            .collect();
        let session = extract_first_header(headers, &names);
        let parent = extract_header_value(headers, PARENT_SESSION_HEADER);
        match self.config.parent_mode {
            SessionParentMode::Group => parent
                .map(|parent_id| format!("parent:{parent_id}"))
                .or_else(|| session.map(str::to_string)),
            SessionParentMode::Independent => session.map(str::to_string),
        }
    }

    fn is_expired(&self, assignment: &StickyAssignment, now: Instant) -> bool {
        now.saturating_duration_since(assignment.last_access) > self.ttl
    }

    fn fresh_index(&self, key: &str, workers: &[Arc<dyn Worker>], now: Instant) -> Option<usize> {
        let assignment = self.table.assignments.get(key)?;
        if self.is_expired(&assignment, now) {
            return None;
        }
        healthy_index(workers, &assignment.worker)
    }

    fn miss_result(&self, key: &str, now: Instant) -> &'static str {
        match self.table.assignments.get(key) {
            None => AFFINITY_NEW,
            Some(assignment) if self.is_expired(&assignment, now) => AFFINITY_EXPIRED,
            Some(_) => AFFINITY_REBIND,
        }
    }

    fn maybe_sweep(&self, now: Instant, keep: Option<&str>) {
        let interval = SWEEP_INTERVAL.min(self.ttl);
        let mut last = self
            .last_sweep
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if now.saturating_duration_since(*last) < interval {
            return;
        }
        *last = now;
        drop(last);
        self.table.retain_live(now, self.ttl, keep);
    }

    fn bump(&self, result: &'static str) {
        self.results[result_index(result)].fetch_add(1, Ordering::Relaxed);
        MeshMetrics::record_session_affinity(result, self.scope.as_str());
    }

    fn allows_soft_move(&self) -> bool {
        self.config.mode == SessionAffinityMode::Soft && self.inner.overrides_session_affinity()
    }

    /// The bound worker at `index` serves the request.
    fn hit(&self, key: &str, index: usize, now: Instant) -> (Option<usize>, RouteDecision) {
        self.table.touch(key, now);
        self.bump(AFFINITY_HIT);
        (Some(index), RouteDecision::branch(AFFINITY_HIT))
    }

    fn record_placement(
        &self,
        key: String,
        worker_index: usize,
        workers: &[Arc<dyn Worker>],
        now: Instant,
        result: &'static str,
    ) {
        let reassigned = (result != AFFINITY_NEW).then_some(result);
        self.table.assign(
            key,
            WorkerIdentity::from_worker(&workers[worker_index]),
            now,
            reassigned,
        );
        self.bump(result);
    }

    async fn commit_soft_move(
        &self,
        key: &str,
        index: usize,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        now: Instant,
    ) -> (Option<usize>, RouteDecision) {
        let mut delegated = info.clone();
        delegated.preferred_worker = Some(index);
        let (chosen, decision) = self.inner.select_with_decision(workers, &delegated).await;
        let Some(chosen) = chosen else {
            return (None, decision);
        };
        let _assign_guard = self.assign_lock.lock().await;
        if chosen != index {
            self.record_placement(key.to_string(), chosen, workers, now, AFFINITY_REBIND);
            return (Some(chosen), decision);
        }
        self.hit(key, chosen, now)
    }
}

fn result_index(result: &str) -> usize {
    match result {
        AFFINITY_HIT => 0,
        AFFINITY_NEW => 1,
        AFFINITY_REBIND => 2,
        AFFINITY_EXPIRED => 3,
        _ => 0,
    }
}

fn healthy_index(workers: &[Arc<dyn Worker>], identity: &WorkerIdentity) -> Option<usize> {
    workers
        .iter()
        .enumerate()
        .find(|(_, worker)| {
            worker.url() == identity.url
                && worker.dp_rank() == identity.dp_rank
                && worker.is_healthy()
                && worker.circuit_breaker().can_execute()
        })
        .map(|(index, _)| index)
}

/// Downcast through the session-affinity wrapper so cache-aware tree seeding
/// still sees the inner policy.
pub(crate) fn as_cache_aware(policy: &dyn LoadBalancingPolicy) -> Option<&super::CacheAwarePolicy> {
    if let Some(wrapper) = policy.as_any().downcast_ref::<SessionAffinityPolicy>() {
        return wrapper.inner().as_any().downcast_ref();
    }
    policy.as_any().downcast_ref()
}

#[async_trait]
impl LoadBalancingPolicy for SessionAffinityPolicy {
    /// A hit reports `hit`. A `dp_sticky` placement reports the affinity result
    /// (`new`, `rebind`, `expired`); any other placement keeps the inner
    /// policy's decision, so its branch and match ratio are still recorded.
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let Some(key) = self.binding_key(info.headers) else {
            return self.inner.select_with_decision(workers, info).await;
        };
        let now = Instant::now();
        self.maybe_sweep(now, Some(&key));
        if let Some(index) = self.fresh_index(&key, workers, now) {
            if !self.allows_soft_move() {
                return self.hit(&key, index, now);
            }
            return self.commit_soft_move(&key, index, workers, info, now).await;
        }

        // Fewest-session placement is synchronous, so the count/choose/insert
        // critical section does not cross an await point.
        if let Some(min_session) = self.inner.as_any().downcast_ref::<MinSessionPolicy>() {
            let _assign_guard = self.assign_lock.lock().await;
            if let Some(index) = self.fresh_index(&key, workers, now) {
                return self.hit(&key, index, now);
            }
            let Some(selected) = min_session.select_fewest(workers) else {
                return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
            };
            let result = self.miss_result(&key, now);
            self.record_placement(key, selected, workers, now, result);
            return (Some(selected), RouteDecision::branch(result));
        }

        let (selected, decision) = self.inner.select_with_decision(workers, info).await;
        let Some(selected) = selected else {
            return (None, decision);
        };
        let _assign_guard = self.assign_lock.lock().await;
        if let Some(index) = self.fresh_index(&key, workers, now) {
            return self.hit(&key, index, now);
        }
        let result = self.miss_result(&key, now);
        self.record_placement(key, selected, workers, now, result);
        (Some(selected), decision)
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        self.inner.on_request_complete(worker_url, success);
    }

    fn on_dispatch(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
    ) {
        self.inner.on_dispatch(pool, headers, worker);
        let Some(key) = self.binding_key(headers) else {
            return;
        };
        if let Some(reason) = self
            .table
            .record_dispatch(&key, WorkerIdentity::from_worker(worker))
        {
            MeshMetrics::record_session_worker_change(pool, reason);
        }
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn needs_request_text(&self) -> bool {
        self.inner.needs_request_text()
    }

    fn needs_tokens(&self) -> bool {
        self.inner.needs_tokens()
    }

    fn wants_tokens(&self) -> bool {
        self.inner.wants_tokens()
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        self.inner.update_loads(loads);
    }

    fn reset(&self) {
        self.inner.reset();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{SessionAffinityMode, SessionParentMode},
        core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType},
    };

    fn workers() -> Vec<Arc<dyn Worker>> {
        ["http://worker-1:8000", "http://worker-2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn header_map(pairs: &[(&str, &str)]) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                http::HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    fn info(headers: &http::HeaderMap) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            headers: Some(headers),
            ..Default::default()
        }
    }

    #[derive(Debug)]
    struct MoveOffPreferred;

    #[async_trait]
    impl LoadBalancingPolicy for MoveOffPreferred {
        async fn select_with_decision(
            &self,
            workers: &[Arc<dyn Worker>],
            info: &SelectWorkerInfo<'_>,
        ) -> (Option<usize>, RouteDecision) {
            let selected = match info.preferred_worker {
                Some(preferred) if workers.len() > 1 => usize::from(preferred == 0),
                Some(preferred) => preferred,
                None => 0,
            };
            (Some(selected), RouteDecision::branch("move_off_preferred"))
        }

        fn overrides_session_affinity(&self) -> bool {
            true
        }

        fn name(&self) -> &'static str {
            "move_off_preferred"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn agentx_default_headers_stick_without_session_id() {
        let policy = SessionAffinityPolicy::dp_sticky(SessionAffinityConfig::default());
        let workers = workers();
        let headers = header_map(&[
            ("x-session-affinity", "corr-1"),
            ("x-correlation-id", "corr-1"),
        ]);
        let info = info(&headers);
        let selected = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(selected));
        assert_eq!(policy.result_count(AFFINITY_NEW), 1);
        assert_eq!(policy.result_count(AFFINITY_HIT), 1);
    }

    #[tokio::test]
    async fn header_priority_prefers_session_id_over_affinity() {
        let policy = SessionAffinityPolicy::dp_sticky(SessionAffinityConfig::default());
        let workers = workers();
        let first = header_map(&[
            ("x-session-id", "session-a"),
            ("x-session-affinity", "session-b"),
        ]);
        let selected = policy.select_worker(&workers, &info(&first)).await.unwrap();

        let affinity_only = header_map(&[("x-session-affinity", "session-b")]);
        let other = policy
            .select_worker(&workers, &info(&affinity_only))
            .await
            .unwrap();
        assert_ne!(other, selected);

        assert_eq!(
            policy.select_worker(&workers, &info(&first)).await,
            Some(selected)
        );
    }

    #[tokio::test]
    async fn empty_higher_priority_header_falls_through() {
        let policy = SessionAffinityPolicy::dp_sticky(SessionAffinityConfig::default());
        let workers = workers();
        let headers = header_map(&[("x-session-id", ""), ("x-session-affinity", "kept")]);
        let info = info(&headers);
        let selected = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(selected));
        assert_eq!(policy.result_count(AFFINITY_HIT), 1);
    }

    #[tokio::test]
    async fn configurable_ttl_expires_and_rebalances() {
        let config = SessionAffinityConfig {
            ttl_secs: 30,
            ..SessionAffinityConfig::default()
        };
        let policy = SessionAffinityPolicy::dp_sticky(config);
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        workers[0].increment_load();
        workers[0].increment_load();
        policy.set_last_access("session-1", Instant::now() - Duration::from_secs(31));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        assert_eq!(policy.result_count(AFFINITY_EXPIRED), 1);
    }

    #[tokio::test]
    async fn correlation_id_alone_does_not_enter_the_table() {
        let policy = SessionAffinityPolicy::dp_sticky(SessionAffinityConfig::default());
        let workers = workers();
        let headers = header_map(&[("x-correlation-id", "req-1")]);
        policy
            .select_worker(&workers, &info(&headers))
            .await
            .unwrap();
        assert_eq!(policy.table_len(), 0);
        assert_eq!(policy.result_count(AFFINITY_NEW), 0);
    }

    #[tokio::test]
    async fn sweep_removes_other_expired_sessions() {
        let config = SessionAffinityConfig {
            ttl_secs: 30,
            ..SessionAffinityConfig::default()
        };
        let policy = SessionAffinityPolicy::dp_sticky(config);
        let workers = workers();
        let session_a = header_map(&[("x-session-id", "session-a")]);
        let session_b = header_map(&[("x-session-id", "session-b")]);
        policy
            .select_worker(&workers, &info(&session_a))
            .await
            .unwrap();
        policy
            .select_worker(&workers, &info(&session_b))
            .await
            .unwrap();
        let expired = Instant::now() - Duration::from_secs(31);
        policy.set_last_access("session-a", expired);
        policy.set_last_access("session-b", expired);
        policy.expire_sweep_clock();

        policy
            .select_worker(&workers, &info(&session_a))
            .await
            .unwrap();
        assert_eq!(policy.result_count(AFFINITY_EXPIRED), 1);
        assert!(policy.has_session("session-a"));
        assert!(!policy.has_session("session-b"));
        assert_eq!(policy.table_len(), 1);
    }

    #[tokio::test]
    async fn hard_mode_keeps_binding_when_inner_would_move() {
        let policy = SessionAffinityPolicy::around(
            SessionAffinityConfig::default(),
            Arc::new(MoveOffPreferred),
        );
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.result_count(AFFINITY_REBIND), 0);
    }

    #[tokio::test]
    async fn wrapped_policy_keeps_its_branch_on_placement_and_counts_worker_changes() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = SessionAffinityPolicy::around(
            SessionAffinityConfig::default(),
            Arc::new(crate::policies::RandomPolicy::new()),
        );
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);

        workers[1].set_healthy(false);
        let (first, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((first, decision.branch), (Some(0), "random"));
        policy.on_dispatch("regular", Some(&headers), &workers[0]);

        workers[1].set_healthy(true);
        let (again, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((again, decision.branch), (Some(0), AFFINITY_HIT));
        policy.on_dispatch("regular", Some(&headers), &workers[0]);

        workers[0].set_healthy(false);
        let (moved, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((moved, decision.branch), (Some(1), "random"));
        policy.on_dispatch("regular", Some(&headers), &workers[1]);

        assert_eq!(policy.result_count(AFFINITY_REBIND), 1);
        let changes: Vec<_> = handle
            .render()
            .lines()
            .filter(|line| line.starts_with("mesh_session_worker_change_total{"))
            .map(str::to_string)
            .collect();
        assert_eq!(
            changes,
            [r#"mesh_session_worker_change_total{pool="regular",reason="rebind"} 1"#]
        );
    }

    #[tokio::test]
    async fn soft_mode_commits_inner_rebind() {
        let config = SessionAffinityConfig {
            mode: SessionAffinityMode::Soft,
            ..SessionAffinityConfig::default()
        };
        let policy = SessionAffinityPolicy::around(config, Arc::new(MoveOffPreferred));
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        assert_eq!(policy.result_count(AFFINITY_REBIND), 1);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
    }

    #[tokio::test]
    async fn group_mode_shares_parent_binding_and_leaves_parent_session_alone() {
        let grouped = SessionAffinityConfig {
            parent_mode: SessionParentMode::Group,
            ..SessionAffinityConfig::default()
        };
        let policy = SessionAffinityPolicy::dp_sticky(grouped);
        let workers = workers();

        let parent = header_map(&[("x-dynamo-session-id", "root")]);
        assert_eq!(
            policy.select_worker(&workers, &info(&parent)).await,
            Some(0)
        );

        let child_a = header_map(&[
            ("x-dynamo-session-id", "child-a"),
            ("x-dynamo-parent-session-id", "root"),
        ]);
        let child_b = header_map(&[
            ("x-dynamo-session-id", "child-b"),
            ("x-dynamo-parent-session-id", "root"),
        ]);
        assert_eq!(
            policy.select_worker(&workers, &info(&child_a)).await,
            Some(1)
        );
        assert_eq!(
            policy.select_worker(&workers, &info(&child_b)).await,
            Some(1)
        );
        assert_eq!(policy.result_count(AFFINITY_HIT), 1);
    }

    #[tokio::test]
    async fn independent_mode_places_children_separately() {
        let policy = SessionAffinityPolicy::dp_sticky(SessionAffinityConfig::default());
        let workers = workers();
        let parent = header_map(&[("x-dynamo-session-id", "root")]);
        assert_eq!(
            policy.select_worker(&workers, &info(&parent)).await,
            Some(0)
        );
        let child_a = header_map(&[
            ("x-dynamo-session-id", "child-a"),
            ("x-dynamo-parent-session-id", "root"),
        ]);
        let child_b = header_map(&[
            ("x-dynamo-session-id", "child-b"),
            ("x-dynamo-parent-session-id", "root"),
        ]);
        assert_eq!(
            policy.select_worker(&workers, &info(&child_a)).await,
            Some(1)
        );
        assert_eq!(
            policy.select_worker(&workers, &info(&child_b)).await,
            Some(0)
        );
    }

    #[tokio::test]
    async fn separate_dp_sticky_tables_do_not_share_bindings() {
        let config = SessionAffinityConfig::default();
        let first = SessionAffinityPolicy::dp_sticky(config.clone());
        let second = SessionAffinityPolicy::dp_sticky(config);
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);
        assert_eq!(first.select_worker(&workers, &info).await, Some(0));
        workers[0].increment_load();
        workers[0].increment_load();
        assert_eq!(first.select_worker(&workers, &info).await, Some(0));
        assert_eq!(second.select_worker(&workers, &info).await, Some(1));
    }

    #[tokio::test]
    async fn wrapped_cache_aware_keeps_its_name_and_downcast() {
        let policy = crate::policies::PolicyFactory::create(
            &crate::config::PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 30,
                max_tree_size: 1000,
            },
            &SessionAffinityConfig::default(),
        );
        assert_eq!(policy.name(), "cache_aware");
        assert!(as_cache_aware(policy.as_ref()).is_some());
    }

    #[tokio::test]
    async fn dp_rank_identity_survives_shared_url() {
        let policy = SessionAffinityPolicy::dp_sticky(SessionAffinityConfig::default());
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|rank| {
                Arc::new(DPAwareWorkerBuilder::new("http://worker:8000", rank, 2).build())
                    as Arc<dyn Worker>
            })
            .collect();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);
        workers[0].increment_load();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        workers[1].increment_load();
        workers[1].increment_load();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
    }
}
