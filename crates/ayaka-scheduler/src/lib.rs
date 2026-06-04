use std::collections::VecDeque;

use ayaka_core::id::BatchId;
use ayaka_request::{RequestResult, RequestState};

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("max_batch_size must be greater than zero")]
    InvalidMaxBatchSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    max_batch_size: usize,
}

impl SchedulerConfig {
    pub fn new(max_batch_size: usize) -> Result<Self, SchedulerError> {
        if max_batch_size == 0 {
            return Err(SchedulerError::InvalidMaxBatchSize);
        }
        Ok(Self { max_batch_size })
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestBatch {
    pub id: BatchId,
    pub requests: Vec<RequestState>,
}

impl RequestBatch {
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn into_requests(self) -> Vec<RequestState> {
        self.requests
    }
}

#[derive(Debug, Clone)]
pub struct SimpleScheduler {
    config: SchedulerConfig,
    queue: VecDeque<RequestState>,
}

impl SimpleScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
        }
    }

    pub fn admit(
        &mut self,
        mut request: RequestState,
    ) -> RequestResult<()> {
        request.mark_queued()?;
        self.queue.push_back(request);
        Ok(())
    }

    pub fn next_batch(&mut self) -> Option<RequestBatch> {
        if self.queue.is_empty() {
            return None;
        }

        let mut requests = Vec::new();
        for _ in 0..self.config.max_batch_size {
            let Some(mut request) = self.queue.pop_front() else {
                break;
            };
            request
                .mark_running()
                .expect("queued requests can transition to running");
            requests.push(request);
        }

        Some(RequestBatch {
            id: BatchId::generate(),
            requests,
        })
    }

    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_core::id::{RequestId, SequenceId};
    use ayaka_request::{RequestState, RequestStatus};

    fn request(id: u64) -> RequestState {
        RequestState::from_parts(
            RequestId::new(id),
            SequenceId::new(id + 100),
            format!("prompt {id}"),
            vec![id as u32],
            2,
            true,
        )
        .unwrap()
    }

    #[test]
    fn fifo_scheduler_batches_up_to_configured_limit() {
        let config = SchedulerConfig::new(2).unwrap();
        assert_eq!(config.max_batch_size(), 2);
        let mut scheduler = SimpleScheduler::new(config);

        scheduler.admit(request(1)).unwrap();
        scheduler.admit(request(2)).unwrap();
        scheduler.admit(request(3)).unwrap();

        let first = scheduler.next_batch().unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first.requests[0].id, RequestId::new(1));
        assert_eq!(first.requests[1].id, RequestId::new(2));
        assert_eq!(first.requests[0].status, RequestStatus::Running);
        assert_eq!(first.requests[1].status, RequestStatus::Running);
        assert_eq!(scheduler.queued_len(), 1);

        let second = scheduler.next_batch().unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second.requests[0].id, RequestId::new(3));
        assert_eq!(second.requests[0].status, RequestStatus::Running);
        assert!(scheduler.next_batch().is_none());
    }

    #[test]
    fn next_batch_returns_none_when_queue_is_empty() {
        let config = SchedulerConfig::new(2).unwrap();
        let mut scheduler = SimpleScheduler::new(config);

        assert!(scheduler.next_batch().is_none());
    }

    #[test]
    fn scheduler_rejects_zero_batch_size() {
        let err = SchedulerConfig::new(0).unwrap_err();
        assert_eq!(err, SchedulerError::InvalidMaxBatchSize);
    }
}
