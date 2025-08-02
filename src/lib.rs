#![no_std]

extern crate alloc;
#[macro_use]
extern crate log;

use axerrno::{LinuxError, LinuxResult};
use axhal::mem::phys_to_virt;
use core::sync::atomic::{AtomicU64, Ordering};
use lru::LruCache;
use memory_addr::{PAGE_SIZE_4K, PhysAddr, VirtAddr};
use spin::Mutex;

pub const PAGE_SHIFT: usize = 12;

#[inline]
fn page_index(offset: u64) -> u64 {
    offset >> PAGE_SHIFT
}

#[inline]
fn page_offset(offset: u64) -> usize {
    (offset & (PAGE_SIZE_4K as u64 - 1)) as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    UpToDate,
    Dirty,
    WriteBack,
    ToWrite,
}

pub trait InodeOps {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> LinuxResult<usize>;
    fn write_at(&self, buf: &[u8], offset: u64) -> LinuxResult<usize>;
    fn len(&self) -> LinuxResult<u64>;
    fn set_len(&self, len: u64) -> LinuxResult;
    fn is_empty(&self) -> LinuxResult<bool> {
        Ok(self.len()? == 0)
    }
}

pub trait PageOps {
    fn alloc_page() -> Option<PhysAddr>;
    fn dealloc_page(addr: PhysAddr);
    fn read_page(addr: VirtAddr, buf: &mut [u8]) -> LinuxResult;
    fn write_page(addr: VirtAddr, buf: &[u8]) -> LinuxResult;
}

#[derive(Debug, Clone, Copy)]
pub struct CachePage {
    pub addr: PhysAddr,
    pub state: PageState,
}

impl CachePage {
    pub fn new(addr: PhysAddr) -> Self {
        Self {
            addr,
            state: PageState::UpToDate,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.state = PageState::Dirty;
    }

    pub fn mark_write_back(&mut self) {
        self.state = PageState::WriteBack;
    }

    pub fn mark_to_write(&mut self) {
        self.state = PageState::ToWrite;
    }

    pub fn mark_up_to_date(&mut self) {
        self.state = PageState::UpToDate;
    }

    pub fn is_dirty(&self) -> bool {
        self.state == PageState::Dirty
    }

    pub fn is_write_back(&self) -> bool {
        self.state == PageState::WriteBack
    }

    pub fn is_to_write(&self) -> bool {
        self.state == PageState::ToWrite
    }

    pub fn is_up_to_date(&self) -> bool {
        self.state == PageState::UpToDate
    }
}

#[derive(Debug)]
pub struct PageCache<N: InodeOps, P: PageOps> {
    pub host: N,
    pages: Mutex<LruCache<u64, CachePage>>,
    file_size: AtomicU64,
    _marker: core::marker::PhantomData<P>,
}

impl<N: InodeOps, P: PageOps> PageCache<N, P> {
    pub fn new(host: N) -> Self {
        let initial_size = host.len().unwrap_or(0);
        Self {
            host,
            pages: Mutex::new(LruCache::unbounded()),
            file_size: AtomicU64::new(initial_size),
            _marker: core::marker::PhantomData,
        }
    }

    pub fn get_page(&self, index: u64) -> Option<CachePage> {
        self.pages.lock().get(&index).copied()
    }

    pub fn load_page(&self, index: u64) -> LinuxResult<CachePage> {
        if let Some(page) = self.get_page(index) {
            trace!("Cache hit: index={}", index);
            return Ok(page);
        }

        trace!("Cache miss: index={}", index);
        let mut page = CachePage::new(P::alloc_page().ok_or(LinuxError::ENOMEM)?);
        let mut buf = [0u8; PAGE_SIZE_4K];
        let offset = index << PAGE_SHIFT;
        let file_len = self.file_size.load(Ordering::Relaxed);

        if offset < file_len {
            let read_size = ((file_len - offset).min(PAGE_SIZE_4K as u64)) as usize;
            self.host.read_at(&mut buf[..read_size], offset)?;
        }

        P::write_page(phys_to_virt(page.addr), &buf)?;
        page.mark_up_to_date();

        self.pages.lock().put(index, page);
        Ok(page)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> LinuxResult<usize> {
        let file_len = self.file_size.load(Ordering::Relaxed);

        if offset >= file_len {
            return Ok(0);
        }

        let read_len = buf.len().min((file_len - offset) as usize);
        let mut current_offset = offset;
        let mut buf_offset = 0;

        while buf_offset < read_len {
            let page_idx = page_index(current_offset);
            let page_off = page_offset(current_offset);
            let remain = PAGE_SIZE_4K - page_off;
            let copy_size = (read_len - buf_offset).min(remain);

            let page = self.load_page(page_idx)?;
            let mut temp_buf = [0u8; PAGE_SIZE_4K];
            P::read_page(phys_to_virt(page.addr), &mut temp_buf)?;

            buf[buf_offset..buf_offset + copy_size]
                .copy_from_slice(&temp_buf[page_off..page_off + copy_size]);

            current_offset += copy_size as u64;
            buf_offset += copy_size;
        }

        Ok(read_len)
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> LinuxResult<usize> {
        let mut current_offset = offset;
        let mut buf_offset = 0;

        while buf_offset < buf.len() {
            let page_idx = page_index(current_offset);
            let page_off = page_offset(current_offset);
            let remain = PAGE_SIZE_4K - page_off;
            let copy_size = (buf.len() - buf_offset).min(remain);

            let page = self.load_page(page_idx)?;

            if page_off != 0 || copy_size < PAGE_SIZE_4K {
                let mut temp_buf = [0u8; PAGE_SIZE_4K];
                P::read_page(phys_to_virt(page.addr), &mut temp_buf)?;
                temp_buf[page_off..page_off + copy_size]
                    .copy_from_slice(&buf[buf_offset..buf_offset + copy_size]);
                P::write_page(phys_to_virt(page.addr), &temp_buf)?;
            } else {
                P::write_page(
                    phys_to_virt(page.addr),
                    &buf[buf_offset..buf_offset + copy_size],
                )?;
            }

            self.pages.lock().get_mut(&page_idx).unwrap().mark_dirty();

            current_offset += copy_size as u64;
            buf_offset += copy_size;
        }

        if !buf.is_empty() {
            self.file_size
                .fetch_max(offset + buf.len() as u64, Ordering::AcqRel);
        }

        Ok(buf.len())
    }

    pub fn write_back_page(&self, index: u64, page: &mut CachePage) -> LinuxResult {
        assert!(page.is_to_write());
        page.mark_write_back();

        let mut buf = [0u8; PAGE_SIZE_4K];
        P::read_page(phys_to_virt(page.addr), &mut buf)?;

        let offset = index << PAGE_SHIFT;
        let file_size = self.file_size.load(Ordering::Relaxed);

        if offset < file_size {
            let write_size = ((file_size - offset).min(PAGE_SIZE_4K as u64)) as usize;
            self.host.write_at(&buf[..write_size], offset)?;
        }

        page.mark_up_to_date();
        Ok(())
    }

    pub fn write_back(&self) -> LinuxResult {
        let file_size = self.file_size.load(Ordering::Relaxed);
        self.host.set_len(file_size)?;

        for (index, page) in self.pages.lock().iter_mut() {
            if page.is_to_write() {
                self.write_back_page(*index, page)?;
            }
        }

        Ok(())
    }

    pub fn sync(&self) -> LinuxResult {
        for (_, page) in self.pages.lock().iter_mut() {
            if page.is_dirty() {
                page.mark_to_write();
            }
        }

        self.write_back()
    }

    pub fn sync_range(&self, start: u64, end: u64) -> LinuxResult {
        let start_index = page_index(start);
        let end_index = page_index(end);

        let mut pages = self.pages.lock();
        for index in start_index..=end_index {
            if let Some(page) = pages.get_mut(&index) {
                if page.is_dirty() {
                    page.mark_to_write();
                }
            }
        }
        drop(pages);

        self.write_back()
    }

    pub fn clear(&self) -> LinuxResult {
        self.sync()?;

        let mut pages = self.pages.lock();
        for (_, page) in pages.iter() {
            P::dealloc_page(page.addr);
        }
        pages.clear();
        Ok(())
    }

    pub fn clear_range(&self, start: u64, end: u64) -> LinuxResult {
        let start_index = page_index(start);
        let end_index = page_index(end);

        let mut pages = self.pages.lock();
        for index in start_index..=end_index {
            if let Some(page) = pages.pop(&index) {
                P::dealloc_page(page.addr);
            }
        }

        Ok(())
    }

    pub fn clear_from_pos(&self, start: u64) -> LinuxResult {
        let start_index = page_index(start);
        let end_index = page_index(self.get_size());

        let mut pages = self.pages.lock();
        for index in start_index..=end_index {
            if let Some(page) = pages.pop(&index) {
                P::dealloc_page(page.addr);
            }
        }
        self.set_size(start);
        Ok(())
    }

    pub fn get_size(&self) -> u64 {
        self.file_size.load(Ordering::Relaxed)
    }

    pub fn set_size(&self, size: u64) {
        self.file_size.store(size, Ordering::Relaxed);
    }
}

