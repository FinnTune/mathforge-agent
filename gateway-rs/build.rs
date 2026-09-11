//! Generates client + server stubs from the shared proto, same source Phase 4's
//! Python side generates from (see scripts/generate_proto.sh at the repo root).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["../proto/chat.proto"], &["../proto"])?;
    Ok(())
}
