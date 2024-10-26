use crate::reader::{ZarrError, ZarrResult};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use itertools::{enumerate, Itertools};
use libc::statx;
use std::ffi::CString;

const _ALIGNMENT: u64 = 512;

// Data to keep track of operations submitted to the queue
struct CompositeUserData {
    file_idx: u32,
    op: u8,
}

impl CompositeUserData {
    fn new(file_idx: u32, op: u8) -> Self {
        Self { file_idx, op }
    }
}

impl From<u64> for CompositeUserData {
    fn from(value: u64) -> Self {
        let file_idx: u32 = (value >> 32)
            .try_into()
            .expect("could not extract file ids from composite user data");
        let op = (value & 0xFF)
            .try_into()
            .expect("could extract operation code from composite user data");
        Self { file_idx, op }
    }
}

impl From<CompositeUserData> for u64 {
    fn from(ud: CompositeUserData) -> u64 {
        let file_idx: u64 = (ud.file_idx as u64) << 32;
        file_idx | ud.op as u64
    }
}

// A struct to keep track of openat + statx operations for a single
// file and organize the results for a read operation.
#[derive(Clone)]
struct OpenFileData {
    fd: Option<types::Fd>,
    file_size: Option<usize>,
    read_size: Option<usize>,
    statx_buf: statx,
}

impl OpenFileData {
    fn new() -> Self {
        Self {
            fd: None,
            file_size: None,
            read_size: None,
            statx_buf: unsafe { std::mem::zeroed() },
        }
    }

    fn get_statx_buffer(&mut self) -> *mut libc::statx {
        &mut self.statx_buf as *mut libc::statx
    }

    fn update(&mut self, cqe_res: i32, code: u8) -> Option<(types::Fd, usize)> {
        match code {
            opcode::OpenAt::CODE => {
                self.fd = Some(types::Fd(cqe_res));
            }
            opcode::Statx::CODE => {
                let s = self.statx_buf.stx_size;
                self.read_size =
                    Some(((s / _ALIGNMENT + u64::from(s % _ALIGNMENT != 0)) * _ALIGNMENT) as usize);
                self.file_size = Some(s as usize);
            }
            _ => panic!("unepected op code when updating open file data"),
        }

        if let (Some(fd), Some(read_size)) = (self.fd, self.read_size) {
            return Some((fd, read_size));
        }

        None
    }

    fn get_fd(&self) -> ZarrResult<types::Fd> {
        if let Some(fd) = self.fd {
            Ok(fd)
        } else {
            Err(ZarrError::Read(
                "Could not retrive file descriptor from open file data".to_string(),
            ))
        }
    }

    fn get_file_size(&self) -> ZarrResult<usize> {
        if let Some(s) = self.file_size {
            Ok(s)
        } else {
            Err(ZarrError::Read(
                "Could not retrive file size from open file data".to_string(),
            ))
        }
    }
}

// A io_uring worker that owns a io_uring and can read a batch
// of files.
struct Worker {
    ring: IoUring<squeue::Entry, cqueue::Entry>,
    files_to_read: Vec<CString>,
    buffers: Option<Vec<Vec<u8>>>,
}

impl Worker {
    fn new(io_uring_size: u32) -> ZarrResult<Self> {
        Ok(Self {
            ring: io_uring::IoUring::builder()
                .setup_sqpoll(1000)
                .build(io_uring_size)?,
            buffers: None,
            files_to_read: Vec::new(),
        })
    }

    fn add_files_to_read(&mut self, filenames: Vec<CString>) {
        self.files_to_read = filenames;
    }

    fn ready_to_run(&self) -> bool {
        !self.files_to_read.is_empty()
    }

    fn get_data(&mut self) -> Vec<Vec<u8>> {
        self.files_to_read = Vec::new();
        self.buffers
            .take()
            .expect("io_uring buffers not instanciated")
    }

    fn run(&mut self) -> ZarrResult<()> {
        let mut n_received_results = 0;
        let mut n_in_flight = 0;
        let mut idx = 0;

        let mut reads_to_push = Vec::with_capacity(self.ring.completion().capacity());
        let mut files_in_flight = vec![OpenFileData::new(); self.files_to_read.len()];
        let mut closes_to_push = Vec::with_capacity(self.ring.completion().capacity());

        let n_files = self.files_to_read.len();
        let mut buffers = vec![Vec::new(); n_files];
        while n_received_results < n_files {
            if idx < n_files && n_in_flight < self.ring.completion().capacity() - 1 {
                let space_left = unsafe {
                    self.ring.submission_shared().capacity() - self.ring.submission_shared().len()
                };

                if space_left > 2 {
                    let open_entry =
                        opcode::OpenAt::new(types::Fd(-1), self.files_to_read[idx].as_ptr())
                            .flags(libc::O_RDONLY | libc::O_DIRECT)
                            .build()
                            .user_data(
                                CompositeUserData::new(idx as u32, opcode::OpenAt::CODE).into(),
                            );
                    let statx_entry = opcode::Statx::new(
                        types::Fd(-1),
                        self.files_to_read[idx].as_ptr(),
                        files_in_flight[idx].get_statx_buffer() as *mut _,
                    )
                    .mask(libc::STATX_SIZE | libc::STATX_DIOALIGN)
                    .build()
                    .user_data(CompositeUserData::new(idx as u32, opcode::Statx::CODE).into());

                    unsafe {
                        self.ring
                            .submission()
                            .push_multiple(&[open_entry, statx_entry])?;
                    }
                    idx += 1;
                    n_in_flight += 2;

                    let l = unsafe {
                        self.ring.submission_shared().len() + self.ring.completion_shared().len()
                    };
                    if l < self.ring.submission().capacity() / 2 {
                        continue;
                    }
                }
            }
            self.ring.submitter().submit()?;

            for cqe in unsafe { self.ring.completion_shared() } {
                let ud: CompositeUserData = cqe.user_data().into();
                let op_idx = ud.file_idx as usize;
                n_in_flight -= 1;
                match ud.op {
                    opcode::OpenAt::CODE | opcode::Statx::CODE => {
                        if let Some((fd, s)) = files_in_flight[op_idx].update(cqe.result(), ud.op) {
                            buffers[op_idx] = vec![0; s as _];
                            let ud = CompositeUserData::new(op_idx as u32, opcode::Read::CODE);
                            let read_entry =
                                opcode::Read::new(fd, buffers[op_idx].as_mut_ptr(), s as _)
                                    .build()
                                    .user_data(ud.into());
                            reads_to_push.push(read_entry);
                        }
                    }
                    opcode::Read::CODE => {
                        let fd = files_in_flight[op_idx].get_fd()?;
                        let s = files_in_flight[op_idx].get_file_size()?;
                        buffers[op_idx] = buffers[op_idx][0..s].to_vec();
                        let ud = CompositeUserData::new(op_idx as u32, opcode::Close::CODE);
                        let close_entry = opcode::Close::new(fd).build().user_data(ud.into());
                        closes_to_push.push(close_entry);
                    }
                    opcode::Close::CODE => {
                        n_received_results += 1;
                    }
                    _ => {
                        panic!("invalid opcode")
                    }
                }
            }

            while !reads_to_push.is_empty()
                && !self.ring.submission().is_full()
                && n_in_flight < self.ring.completion().capacity()
            {
                let entry = reads_to_push
                    .pop()
                    .expect("read entries to push to queue should not be empty");
                unsafe {
                    self.ring.submission().push(&entry)?;
                }
                n_in_flight += 1;
            }

            while !closes_to_push.is_empty()
                && !self.ring.submission().is_full()
                && n_in_flight < self.ring.completion().capacity()
            {
                let entry = closes_to_push
                    .pop()
                    .expect("close entries to push to queue should not be empty");
                unsafe {
                    self.ring.submission().push(&entry)?;
                }
                n_in_flight += 1;
            }

            self.ring.submitter().submit()?;
        }
        self.buffers = Some(buffers);

        Ok(())
    }
}

// A pool of io uring workers
pub struct WorkerPool {
    workers: Vec<Worker>,
    pool: rayon::ThreadPool,
    filenames: Option<Vec<CString>>,
}

impl WorkerPool {
    pub fn new(ring_size: u32, n_workers: usize) -> ZarrResult<Self> {
        let mut workers = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            workers.push(Worker::new(ring_size)?);
        }
        Ok(Self {
            workers,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(n_workers)
                .build()?,
            filenames: None,
        })
    }

    pub fn run(&mut self) -> ZarrResult<()> {
        if let Some(filenames) = &self.filenames {
            let chunk_size = filenames.len() / self.workers.len()
                + usize::from(filenames.len() % self.workers.len() != 0);
            for (idx, files_chunk) in enumerate(
                self.filenames
                    .take()
                    .expect("filenames unexpectedly not instanciated")
                    .chunks(chunk_size),
            ) {
                self.workers[idx].add_files_to_read(files_chunk.to_vec());
            }

            self.pool.scope(|s| {
                for worker in self.workers.split_inclusive_mut(|_| true) {
                    s.spawn(move |_| {
                        if worker[0].ready_to_run() {
                            worker[0].run().expect("io uring worker failed to run");
                        }
                    });
                }
            })
        } else {
            return Err(ZarrError::Read(
                "filenames not instanciated in io uring worker pool".to_string(),
            ));
        }
        Ok(())
    }

    pub fn add_file(&mut self, filename: String) -> ZarrResult<()> {
        let filename = CString::new(filename)?;
        if let Some(filenames) = &mut self.filenames {
            filenames.push(filename);
        } else {
            self.filenames = Some(vec![filename]);
        }

        Ok(())
    }

    pub fn get_data(&mut self) -> Vec<Vec<u8>> {
        let vecs = self
            .workers
            .iter_mut()
            .map(|w| {
                if w.ready_to_run() {
                    w.get_data()
                } else {
                    Vec::new()
                }
            })
            .filter(|v| !v.is_empty())
            .collect_vec();
        itertools::concat(vecs)
    }
}

// TODO add tests for the worker pool
#[cfg(test)]
mod io_uring_tests {}
