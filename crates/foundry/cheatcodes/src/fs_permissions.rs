//! Support for controlling fs access

use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Configures file system access
///
/// E.g. for cheat codes (`vm.writeFile`)
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FsPermissions {
    /// what kind of access is allowed
    pub permissions: Vec<PathPermission>,
}

// === impl FsPermissions ===

impl FsPermissions {
    /// Creates anew instance with the given `permissions`
    pub fn new(permissions: impl IntoIterator<Item = PathPermission>) -> Self {
        Self {
            permissions: permissions.into_iter().collect(),
        }
    }

    /// Adds a new permission
    pub fn add(&mut self, permission: PathPermission) {
        self.permissions.push(permission);
    }

    /// Returns true if access to the specified path is allowed with the
    /// specified.
    ///
    /// This first checks permission, and only if it is granted, whether the
    /// path is allowed.
    ///
    /// We only allow paths that are inside  allowed paths.
    ///
    /// Caution: This should be called with normalized paths if the
    /// `allowed_paths` are also normalized.
    pub fn is_path_allowed(&self, path: &Path, kind: FsAccessKind) -> bool {
        self.find_permission(path)
            .map(|perm| perm.is_granted(kind))
            .unwrap_or_default()
    }

    /// Returns the permission for the matching path.
    ///
    /// This finds the longest matching path with resolved sym links, e.g. if we
    /// have the following permissions:
    ///
    /// `./out` = `read`
    /// `./out/contracts` = `read-write`
    ///
    /// And we check for `./out/contracts/MyContract.sol` we will get
    /// `read-write` as permission.
    pub fn find_permission(&self, path: &Path) -> Option<FsAccessPermission> {
        let mut permission: Option<&PathPermission> = None;
        for perm in &self.permissions {
            let permission_path = dunce::canonicalize(&perm.path).unwrap_or(perm.path.clone());
            if path.starts_with(permission_path) {
                if let Some(active_perm) = permission.as_ref() {
                    // the longest path takes precedence
                    if perm.path < active_perm.path {
                        continue;
                    }
                }
                permission = Some(perm);
            }
        }
        permission.map(|perm| perm.access)
    }

    /// Updates all `allowed_paths` and joins ([`Path::join`]) the `root` with
    /// all entries
    pub fn join_all(&mut self, root: impl AsRef<Path>) {
        let root = root.as_ref();
        self.permissions.iter_mut().for_each(|perm| {
            perm.path = root.join(&perm.path);
        });
    }

    /// Same as [`Self::join_all`] but consumes the type
    pub fn joined(mut self, root: impl AsRef<Path>) -> Self {
        self.join_all(root);
        self
    }

    /// Removes all existing permissions for the given path
    pub fn remove(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        self.permissions
            .retain(|permission| permission.path != path);
    }

    /// Returns true if no permissions are configured
    pub fn is_empty(&self) -> bool {
        self.permissions.is_empty()
    }

    /// Returns the number of configured permissions
    pub fn len(&self) -> usize {
        self.permissions.len()
    }
}

/// Represents an access permission to a single path
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathPermission {
    /// Permission level to access the `path`
    pub access: FsAccessPermission,
    /// The targeted path guarded by the permission
    pub path: PathBuf,
}

// === impl PathPermission ===

impl PathPermission {
    /// Returns a new permission for the path and the given access
    pub fn new(path: impl Into<PathBuf>, access: FsAccessPermission) -> Self {
        Self {
            path: path.into(),
            access,
        }
    }

    /// Returns a new read-only permission for the path
    pub fn read(path: impl Into<PathBuf>) -> Self {
        Self::new(path, FsAccessPermission::Read)
    }

    /// Returns a new read-write permission for the path
    pub fn read_write(path: impl Into<PathBuf>) -> Self {
        Self::new(path, FsAccessPermission::ReadWrite)
    }

    /// Returns a new write-only permission for the path
    pub fn write(path: impl Into<PathBuf>) -> Self {
        Self::new(path, FsAccessPermission::Write)
    }

    /// Returns a non permission for the path
    pub fn none(path: impl Into<PathBuf>) -> Self {
        Self::new(path, FsAccessPermission::None)
    }

    /// Returns true if the access is allowed
    pub fn is_granted(&self, kind: FsAccessKind) -> bool {
        self.access.is_granted(kind)
    }
}

/// Represents the operation on the fs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsAccessKind {
    /// read from fs (`vm.readFile`)
    Read,
    /// write to fs (`vm.writeFile`)
    Write,
}

impl fmt::Display for FsAccessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FsAccessKind::Read => f.write_str("read"),
            FsAccessKind::Write => f.write_str("write"),
        }
    }
}

/// Determines the status of file system access
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FsAccessPermission {
    /// FS access is _not_ allowed
    #[default]
    None,
    /// FS access is allowed, this includes `read` + `write`
    ReadWrite,
    /// Only reading is allowed
    Read,
    /// Only writing is allowed
    Write,
}

// === impl FsAccessPermission ===

impl FsAccessPermission {
    /// Returns true if the access is allowed
    pub fn is_granted(&self, kind: FsAccessKind) -> bool {
        #[allow(clippy::match_same_arms)]
        match (self, kind) {
            (FsAccessPermission::ReadWrite, _) => true,
            (FsAccessPermission::None, _) => false,
            (FsAccessPermission::Read, FsAccessKind::Read) => true,
            (FsAccessPermission::Write, FsAccessKind::Write) => true,
            _ => false,
        }
    }
}

impl FromStr for FsAccessPermission {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "true" | "read-write" | "readwrite" => Ok(FsAccessPermission::ReadWrite),
            "false" | "none" => Ok(FsAccessPermission::None),
            "read" => Ok(FsAccessPermission::Read),
            "write" => Ok(FsAccessPermission::Write),
            _ => Err(format!("Unknown variant {s}")),
        }
    }
}

impl fmt::Display for FsAccessPermission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FsAccessPermission::ReadWrite => f.write_str("read-write"),
            FsAccessPermission::None => f.write_str("none"),
            FsAccessPermission::Read => f.write_str("read"),
            FsAccessPermission::Write => f.write_str("write"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_parse_permission() {
        assert_eq!(FsAccessPermission::ReadWrite, "true".parse().unwrap());
        assert_eq!(FsAccessPermission::ReadWrite, "readwrite".parse().unwrap());
        assert_eq!(FsAccessPermission::ReadWrite, "read-write".parse().unwrap());
        assert_eq!(FsAccessPermission::None, "false".parse().unwrap());
        assert_eq!(FsAccessPermission::None, "none".parse().unwrap());
        assert_eq!(FsAccessPermission::Read, "read".parse().unwrap());
        assert_eq!(FsAccessPermission::Write, "write".parse().unwrap());
    }

    #[test]
    fn nested_permissions() {
        let permissions = FsPermissions::new(vec![
            PathPermission::read("./"),
            PathPermission::write("./out"),
            PathPermission::read_write("./out/contracts"),
        ]);

        let permission = permissions
            .find_permission(Path::new("./out/contracts/MyContract.sol"))
            .unwrap();
        assert_eq!(FsAccessPermission::ReadWrite, permission);
        let permission = permissions
            .find_permission(Path::new("./out/MyContract.sol"))
            .unwrap();
        assert_eq!(FsAccessPermission::Write, permission);
    }
}
