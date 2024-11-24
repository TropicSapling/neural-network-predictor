// All I/O is upscaled/downscaled by 128x
pub const RESOLUTION: f64 = 128.0;

// Partitions for cross-validation
pub const PARTITIONS: usize = 2;

// Array sizes
pub const DATA_SIZE: usize = (TEST_SIZE + INPS_SIZE) * (PARTITIONS + 2);
pub const INPS_SIZE: usize = INPS/OUTS;
pub const TEST_SIZE: usize = 16;

// I/O
pub const INPS: usize = 32;
pub const OUTS: usize = 2;