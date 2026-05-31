# WasmStone demo app

This project is a PoC of a simple image classifier in Wasm, demonstrating the feasibility of running an AI confidentially inside a Keystone enclave.

## Development environment

1. Update and set your rustup environment to `nightly` :

```sh
rustup self update
rustup default nightly
```

2. Add a new target for WasiP2 :

```sh
rustup target add wasm32-wasip2
```

## Compilation

1. Compile to wasip2

```sh
cargo build --target wasm32-wasip2 --release
```

 2. Compile for Pulley. This needs to be done with the same Wasmtime version as the one that is running inside the enclave - which currently is version `33.0.1`.
 ```sh
wasmtime compile \
         --target pulley64 \
         -W max-memory-size=0x10000000 \
         -O memory-reservation=0x10000000 \
         target/wasm32-wasip2/release/classifier.wasm
 ```

## Run

### Wasmtime

Tested with `wasmtime 33.0.1`.

You need to have the following files :

```
.
├── classifier.cwasm
├── mnist.onnx
└── wasi_config.toml
```

Run the binary:

```sh
wasmtime run  --allow-precompiled --dir . --config=wasi_config.toml target/wasm32-wasip2/release/classifier.wasm
```

The classifier has a REST API and listens to port `:3000`:

```sh
curl http://localhost:3000/

<!DOCTYPE html>
        <html>
        <body>
            <h1>Upload PNG Image</h1>
            <form action="/image" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/png"/>
                <input type="submit" value="Upload"/>
            </form>
        </body>
        </html
```

## Credits

The `tokio` crate present in this repository is inspired from [this PR](https://github.com/tokio-rs/tokio/pull/6893) on which we made additional changes.
