use crate::errors::zarr_errors::ZarrQueryResult;
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::Notify;

/// More or less copied from here, https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/thread_pools.rs
/// with a few tweaks to make this a runtime for non blockinng i/o.
pub(crate) struct IoRuntime {
    /// Handle is the tokio structure for interacting with a Runtime.
    handle: Handle,
    /// Signal to start shutting down.
    notify_shutdown: Arc<Notify>,
    /// When thread is active, is Some.
    thread_join_handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for IoRuntime {
    fn drop(&mut self) {
        // Notify the thread to shutdown.
        self.notify_shutdown.notify_one();

        // TODO make sure that no tasks can be added to the runtime
        // past this point.

        if let Some(thread_join_handle) = self.thread_join_handle.take() {
            // If the thread is still running, we wait for it to finish.
            if let Err(e) = thread_join_handle.join() {
                eprintln!("Error joining CPU runtime thread: {e:?}",);
            }
        }
    }
}

impl IoRuntime {
    /// Create a new Tokio Runtime for non-blocking tasks.
    pub(crate) fn try_new() -> ZarrQueryResult<Self> {
        // So the way I'm thinking about this is I'm going to submit one
        // i/o task at a time, the point is for that task, which will be
        // reading a chunk, to run concurrently with decoding the
        // previously read chunk. So for now, the runtime will have
        // a single thread, since I don't need more than that.
        let io_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .enable_io()
            .build()?;
        let handle = io_runtime.handle().clone();
        let notify_shutdown = Arc::new(Notify::new());
        let notify_shutdown_captured = Arc::clone(&notify_shutdown);

        // The io_runtime runs and is dropped on a separate thread.
        let thread_join_handle = std::thread::spawn(move || {
            io_runtime.block_on(async move {
                notify_shutdown_captured.notified().await;
            });
            // The io_runtime is dropped here, which will wait for all tasks
            // to complete.
        });

        Ok(Self {
            handle,
            notify_shutdown,
            thread_join_handle: Some(thread_join_handle),
        })
    }

    pub(crate) fn handle(&self) -> &Handle {
        &self.handle
    }
}
