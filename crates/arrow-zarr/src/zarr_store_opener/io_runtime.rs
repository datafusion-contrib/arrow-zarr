// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use tokio::runtime::Handle;
use tokio::sync::Notify;

use super::zarr_errors::ZarrQueryResult;

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
        let io_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
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
