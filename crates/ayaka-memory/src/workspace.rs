use std::collections::HashMap;

use crate::{DeviceSpan, MemoryError, Result};

#[cfg(feature = "cuda")]
use ayaka_core::device::Device;
#[cfg(feature = "cuda")]
use crate::{DeviceBuffer, MemoryPurpose};

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SectionId(u32);

impl SectionId {
    pub const fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone)]
struct SectionRequirement {
    name: String,
    max_bytes: usize,
    align: usize,
}

#[derive(Debug, Clone)]
pub struct SectionInfo {
    pub id: SectionId,
    pub name: String,
    pub span: DeviceSpan,
}

#[derive(Debug, Default)]
pub struct WorkspacePlan {
    requirements: Vec<SectionRequirement>,
    names: HashMap<String, SectionId>,
}

impl WorkspacePlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn require(&mut self, name: impl Into<String>, max_bytes: usize, align: usize) -> Result<SectionId> {
        let name = name.into();
        if max_bytes == 0 {
            return Err(MemoryError::invalid(format!(
                "workspace section {name} max_bytes must be non-zero"
            )));
        }
        if align == 0 || !align.is_power_of_two() {
            return Err(MemoryError::invalid(format!(
                "workspace section {name} alignment must be a non-zero power of two"
            )));
        }
        if self.names.contains_key(&name) {
            return Err(MemoryError::WorkspaceDuplicateSection { name });
        }
        let id = SectionId(self.requirements.len() as u32);
        self.requirements.push(SectionRequirement {
            name: name.clone(),
            max_bytes,
            align,
        });
        self.names.insert(name, id);
        Ok(id)
    }

    pub fn required_bytes(&self) -> Result<usize> {
        let mut cursor = 0usize;
        for req in &self.requirements {
            cursor = align_up(cursor, req.align)?;
            cursor = cursor
                .checked_add(req.max_bytes)
                .ok_or_else(|| MemoryError::overflow("workspace required byte count overflow"))?;
        }
        Ok(cursor)
    }

    pub fn freeze_with_span(self, base: DeviceSpan) -> Result<Workspace> {
        let required = self.required_bytes()?;
        if required > base.len {
            return Err(MemoryError::invalid(format!(
                "workspace requires {required} bytes but base span has {} bytes",
                base.len
            )));
        }

        let mut cursor = 0usize;
        let mut sections = Vec::with_capacity(self.requirements.len());
        for (index, req) in self.requirements.into_iter().enumerate() {
            cursor = align_up(cursor, req.align)?;
            let span = base.checked_subspan(cursor, req.max_bytes)?;
            cursor += req.max_bytes;
            sections.push(SectionInfo {
                id: SectionId(index as u32),
                name: req.name,
                span,
            });
        }
        Ok(Workspace { sections })
    }

    #[cfg(feature = "cuda")]
    pub fn freeze(self, device: Device) -> Result<OwnedWorkspace> {
        let required = self.required_bytes()?;
        let buffer = DeviceBuffer::reserve_for(device, required, MemoryPurpose::Workspace)?;
        let workspace = self.freeze_with_span(buffer.span())?;
        Ok(OwnedWorkspace { buffer, workspace })
    }
}

#[derive(Debug, Clone)]
pub struct Workspace {
    sections: Vec<SectionInfo>,
}

impl Workspace {
    pub fn section(&self, id: SectionId) -> Result<DeviceSpan> {
        self.sections
            .get(id.raw() as usize)
            .map(|info| info.span)
            .ok_or(MemoryError::WorkspaceUnknownSection { id: id.raw() })
    }

    pub fn info(&self, id: SectionId) -> Result<&SectionInfo> {
        self.sections
            .get(id.raw() as usize)
            .ok_or(MemoryError::WorkspaceUnknownSection { id: id.raw() })
    }

    pub fn sections(&self) -> &[SectionInfo] {
        &self.sections
    }
}

#[cfg(feature = "cuda")]
pub struct OwnedWorkspace {
    buffer: DeviceBuffer,
    workspace: Workspace,
}

#[cfg(feature = "cuda")]
impl OwnedWorkspace {
    pub fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub fn section(&self, id: SectionId) -> Result<DeviceSpan> {
        self.workspace.section(id)
    }
}

#[cfg(not(feature = "cuda"))]
pub struct OwnedWorkspace;

fn align_up(value: usize, align: usize) -> Result<usize> {
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| MemoryError::overflow("workspace align_up overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freeze_assigns_stable_aligned_sections() {
        let mut plan = WorkspacePlan::new();
        let a = plan.require("attn_plan", 100, 64).unwrap();
        let b = plan.require("decode_tmp", 32, 256).unwrap();
        let workspace = plan.freeze_with_span(DeviceSpan::new(0x1000, 1024)).unwrap();
        assert_eq!(workspace.section(a).unwrap(), DeviceSpan::new(0x1000, 100));
        assert_eq!(workspace.section(b).unwrap().ptr, 0x1100);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut plan = WorkspacePlan::new();
        plan.require("attn_plan", 100, 64).unwrap();
        assert!(plan.require("attn_plan", 200, 64).is_err());
    }

    #[test]
    fn required_bytes_includes_alignment_padding() {
        let mut plan = WorkspacePlan::new();
        plan.require("a", 100, 64).unwrap();
        plan.require("b", 32, 256).unwrap();
        assert_eq!(plan.required_bytes().unwrap(), 288);
    }
}
