# SAM2 GPU streaming pipeline

The pipeline keeps two reusable pinned-host slots and two reusable GPU slots.
Inference owns one slot while the staging thread prepares the following frame in
the other slot. No per-frame pinned or device allocation is required.

```mermaid
sequenceDiagram
    participant D as Decoder thread
    participant Q as Pageable CPU queue
    participant S as Staging thread
    participant P as 2 pinned slots
    participant T as CUDA transfer stream
    participant G as 2 GPU slots
    participant I as CUDA inference stream
    participant M as SAM2

    par Prepare frame n
        D->>Q: Decode and normalize frame n
        Q->>S: Pop frame n
        S->>P: Copy into pinned slot A
        S->>T: Enqueue H2D to GPU slot A
        T-->>I: Record copy_ready[n]
    and Decode frame n+1
        D->>Q: Decode and normalize frame n+1
    end

    I->>I: Wait on copy_ready[n]
    I->>M: Run frame n inference

    par Inference for frame n
        M-->>I: Image encoder, memory attention, heads, memory encoder
    and Prepare frame n+1
        Q->>S: Pop frame n+1
        S->>P: Copy into pinned slot B
        S->>T: Enqueue H2D to GPU slot B
        T-->>I: Record copy_ready[n+1]
    end

    I-->>S: Record consumed[n]
    S->>S: Reuse slot A only after consumed[n]
    I->>I: Wait on copy_ready[n+1]
    I->>M: Run frame n+1 inference
```

## Ownership rules

- The decoder owns pageable tensors until the staging thread copies them.
- The staging thread is the only writer to pinned and GPU staging slots.
- The inference stream waits on `copy_ready[frame]` before reading a GPU slot.
- A frame lease remains active for the complete inference step.
- The staging thread waits on `consumed[frame]` before recycling that slot.
- Queue capacity, pinned allocation and GPU allocation are fixed independently of
  video length.

The standalone timeline is available in
[`SAM2_GPU_STREAMING_PIPELINE.svg`](SAM2_GPU_STREAMING_PIPELINE.svg).
